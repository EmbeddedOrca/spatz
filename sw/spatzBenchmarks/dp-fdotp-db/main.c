// Copyright 2022 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matteo Perotti <mperotti@iis.ee.ethz.ch>

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/fdotp-db.c"

// #define TIMING 1

#define DBG_LVL_NONE 0
#define DBG_LVL_ERR 1
#define DBG_LVL_WARN 2
#define DBG_LVL_INFO 3
#define DBG_LVL_DBG 4

#define DBG_LVL DBG_LVL_NONE

#define DEBUG(func, lvl) \
  if (lvl <= DBG_LVL) { \
    func; \
  }

#define MIN(a, b) ((a) < (b) ? (a) : (b))

double *l1_buf;
double *a;
double *b;
double *result_ma;
double *result_simple;

double result[2] = { 0 };

static inline int fp_check(const double a, const double b) {
  const double threshold = 0.00001;

  // Absolute value
  double comp = a - b;
  if (comp < 0)
    comp = -comp;

  return comp > threshold;
}

double fdotp(double *a, double *b, unsigned int num) {
  double acc = a[0] * b[0];
  double res = 0.0;

  for (unsigned int i = 1; i < num; ++i) {
    res = a[i] * b[i];
    acc += res;
    DEBUG(printf("a[%u] = %f, b[%u] = %f, acc = %f\n", i, a[i], i, b[i], acc), DBG_LVL_DBG);
    snrt_cluster_hw_barrier();
  }

  return acc;
}

double vreduce_sum(double *a, unsigned int avl) {
  const unsigned int orig_avl = avl;
  unsigned int vl;
  double res;

  asm volatile("vsetvli %0, %1, e64, m8, ta, ma" : "=r"(vl) : "r"(avl));
  asm volatile("vmv.s.x v0, zero");

  do {
    asm volatile("vsetvli %0, %1, e64, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile("vle64.v v8, (%0)" ::"r"(a));
    asm volatile("vfredusum.vs v0, v8, v0");

    DEBUG(printf("vl = %u, avl = %u\n", vl, avl), DBG_LVL_DBG);

    a += vl;
    avl -= vl;
  } while (avl > 0);

  asm volatile("vsetvli zero, %0, e64, m8, ta, ma" ::"r"(orig_avl));
  asm volatile("vfmv.f.s %0, v0" : "=f"(res));

  return res;
}

/**
 * @brief Implementation of the dot product using memory-aware double buffering.
 *
 * This function computes the dot product of two vectors A and B of size `dim`
 */
double dp_dotp_db_ma(
  double *dotp_A_dram,
  double *dotp_B_dram,
  unsigned int vec_dim,

  double *l1_buf,
  unsigned int l1_buf_len
) {
  const unsigned int cid = snrt_cluster_core_idx();
  const unsigned int num_cores = snrt_cluster_core_num();

  // HW and algo configuration
  const unsigned int CHUNK_SIZE = 8; // Half of L1 banks
  const unsigned int T_S = sizeof(double);

  if (l1_buf_len % 32 != 0) {
    printf("L1 buffer length must be a multiple of the number of banks times the number of cores (32).\n");
    return 0.0;
  }

  // Number of element per iteration
  const unsigned MAX_NUM_ELEM_BUF = l1_buf_len / 4;
  const unsigned int NUM_ELEM_PER_ITER = MIN(MAX_NUM_ELEM_BUF, vec_dim / 2);

  // Pointers to the left and right side of L1 memory
  double *a0 = l1_buf;
  double *b0 = l1_buf + 2 * NUM_ELEM_PER_ITER;
  double *a1 = a0 + CHUNK_SIZE;
  double *b1 = b0 + CHUNK_SIZE;
  double *temp = NULL;
  double res = 0.0;
  result[cid] = 0.0;

  // Pointer increment
  const unsigned int NUM_ELEM_PER_CORE = NUM_ELEM_PER_ITER / num_cores;
  const unsigned int CORE_OFFSET = cid * NUM_ELEM_PER_ITER;
  const unsigned int DMA2D_REPEAT = NUM_ELEM_PER_ITER / CHUNK_SIZE;

  if (cid == 0) {
    printf("a0: %p, b0: %p, a1: %p, b1: %p\n", a0, b0, a1, b1);
    printf("MAX_NUM_ELEM_BUF: %u\n", MAX_NUM_ELEM_BUF);
    printf("NUM_ELEM_PER_ITER: %u\n", NUM_ELEM_PER_ITER);
    printf("2D DMA REPEAT: %u\n", DMA2D_REPEAT);
  }

  // Timing
  unsigned int load_timer = (unsigned int)-1;
  unsigned int calc_timer = (unsigned int)-1;
  unsigned int calc_timer_avg = 0;
  unsigned dma_wait_tot = 0;
  unsigned int performance_timer = 0;

  // Current element to load from DRAM
  unsigned int load_idx = 0;

  // Start the initial DMA transfer
  if (cid == 0) {
    DEBUG(printf("Load index: %u, vec_dim: %u\n", load_idx, vec_dim), DBG_LVL_DBG);
    snrt_dma_start_2d(
      a0,
      dotp_A_dram + load_idx,
      CHUNK_SIZE * T_S,
      2 * CHUNK_SIZE * T_S,
      CHUNK_SIZE * T_S,
      DMA2D_REPEAT
    );
    snrt_dma_start_2d(
      b0,
      dotp_B_dram + load_idx,
      CHUNK_SIZE * T_S,
      2 * CHUNK_SIZE * T_S,
      CHUNK_SIZE * T_S,
      DMA2D_REPEAT
    );
    performance_timer = benchmark_get_cycle();
  }

  snrt_cluster_hw_barrier();

  do {

    // Wait for the data load of the current calc data
    if (cid == 0) {
#ifdef TIMING
      unsigned dma_wait = benchmark_get_cycle();
#endif
      snrt_dma_wait_all();
#ifdef TIMING
      dma_wait = benchmark_get_cycle() - dma_wait;
      dma_wait_tot += dma_wait;
#endif
    }

    load_idx += NUM_ELEM_PER_ITER;

    snrt_cluster_hw_barrier();

    // Start the DMA transfer on chunk i + 1
    if (cid == 0) {
      if (load_idx < vec_dim) {
        DEBUG(printf("Load index: %u, vec_dim: %u\n", load_idx, vec_dim), DBG_LVL_DBG);
        snrt_dma_start_2d(
          a1,
          dotp_A_dram + load_idx,
          CHUNK_SIZE * T_S,
          2 * CHUNK_SIZE * T_S,
          CHUNK_SIZE * T_S,
          DMA2D_REPEAT
        );
        snrt_dma_start_2d(
          b1,
          dotp_B_dram + load_idx,
          CHUNK_SIZE * T_S,
          2 * CHUNK_SIZE * T_S,
          CHUNK_SIZE * T_S,
          DMA2D_REPEAT
        );
      }
    }

#ifdef TIMING
    if (cid == 0) {
      calc_timer = benchmark_get_cycle();
    }
#endif

    DEBUG(printf("Core %u - Starting fdotp_v64b_ma with a0: %p, b0: %p, CORE_OFFSET: %u, NUM_ELEM_PER_CORE: %u\n",
           cid, a0 + CORE_OFFSET, b0 + CORE_OFFSET,
           CORE_OFFSET, NUM_ELEM_PER_CORE), DBG_LVL_DBG);
    result[cid] = fdotp_v64b_ma(
      a0 + CORE_OFFSET,
      b0 + CORE_OFFSET,
      NUM_ELEM_PER_CORE,
      result[cid]
    );

#ifdef TIMING
    if (cid == 0) {
      calc_timer = benchmark_get_cycle() - calc_timer;
      calc_timer_avg += calc_timer;
    }
#endif

  // Swap the pointers
    temp = a0;
    a0 = a1;
    a1 = temp;

    temp = b0;
    b0 = b1;
    b1 = temp;

  } while (load_idx < vec_dim);

  snrt_cluster_hw_barrier();

  if (cid == 0) {
    performance_timer = benchmark_get_cycle() - performance_timer;
  }

#ifdef TIMING
  if (cid == 0) {
    printf("The DMA wait took %u cycles on average.\n", dma_wait_tot / iter);
    printf("The calculation took %u cycles on average.\n", calc_timer_avg / iter);
  }
#endif

  // End dump, Record the time
  if (cid == 0) {
    long unsigned int performance = 1000 * 2 * vec_dim / performance_timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE);

    printf("\n----- (%d) dp fdotp MA -----\n", vec_dim);
    printf("The calculation took %u cycles.\n", performance_timer);
    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }

  snrt_cluster_hw_barrier();

  return res;
}

double dp_dotp_db_simple(
  double *dotp_A_dram,
  double *dotp_B_dram,
  unsigned int vec_dim,

  double *l1_buf,
  unsigned int l1_buf_len // In sizeof doubles
) {
  const unsigned int cid = snrt_cluster_core_idx();
  const unsigned int num_cores = snrt_cluster_core_num();

  const unsigned int T_S = sizeof(double);

  if (l1_buf_len % 32 != 0) {
    printf("L1 buffer length must be a multiple of the number of banks times the number of cores (32).\n");
    return 0.0;
  }

  // Split the buffer into left and right side.
  double *a0 = l1_buf;
  double *b0 = l1_buf + (l1_buf_len / 4);
  double *a1 = l1_buf + (l1_buf_len / 2);
  double *b1 = l1_buf + (3 * l1_buf_len / 4);
  double *temp = NULL;
  double res = 0.0;
  result[cid] = 0.0;

  // Pointer increment
  const unsigned int NUM_ELEM_PER_ITER = b0 - a0 > vec_dim / 2 ? vec_dim / 2 : b0 - a0;
  const unsigned int NUM_ELEM_PER_CORE = NUM_ELEM_PER_ITER / num_cores;
  const unsigned int CORE_OFFSET = cid * NUM_ELEM_PER_CORE;
  printf("NUM_ELEM_PER_ITER: %u\n", NUM_ELEM_PER_ITER);

  // Timing variables
  unsigned int calc_timer = 0;
  unsigned int calc_timer_avg = 0;
  unsigned dma_wait_tot = 0;
  unsigned int performance_timer = 0;

  // Which elements to load next in doubles
  unsigned int load_idx = 0;

  if (cid == 0) {
    snrt_dma_start_1d(
      a0,
      dotp_A_dram + load_idx, // load index is in size of type
      NUM_ELEM_PER_ITER * T_S // in bytes
    );
    snrt_dma_start_1d(
      b0,
      dotp_B_dram + load_idx,
      NUM_ELEM_PER_ITER * T_S
    );
    performance_timer = benchmark_get_cycle();
  }

  snrt_cluster_hw_barrier();

  do {
    // Wait for the data load of the current calc data
    if (cid == 0) {
#ifdef TIMING
      unsigned dma_wait = benchmark_get_cycle();
#endif
      snrt_dma_wait_all();
#ifdef TIMING
      dma_wait = benchmark_get_cycle() - dma_wait;
      dma_wait_tot += dma_wait;
#endif
    }

    load_idx += NUM_ELEM_PER_ITER; // increment the index of the next load

    snrt_cluster_hw_barrier();

    // Load the i + 1 subvector
    if (cid == 0) {
      if (load_idx < vec_dim) {
        snrt_dma_start_1d(
          a1,
          dotp_A_dram + load_idx,
          NUM_ELEM_PER_ITER * T_S
        );
        snrt_dma_start_1d(
          b1,
          dotp_B_dram + load_idx,
          NUM_ELEM_PER_ITER * T_S
        );
      }
    }

    // Calculate the i sub-vector dot product
#ifdef TIMING
    if (cid == 0) {
      calc_timer = benchmark_get_cycle();
    }
#endif

    result[cid] = fdotp_v64b(
      a0 + CORE_OFFSET,
      b0 + CORE_OFFSET,
      NUM_ELEM_PER_CORE,
      result[cid]
    );

#ifdef TIMING
    if (cid == 0) {
      calc_timer = benchmark_get_cycle() - calc_timer;
      calc_timer_avg += calc_timer;
    }
#endif

  // Swap the pointers
    temp = a0;
    a0 = a1;
    a1 = temp;

    temp = b0;
    b0 = b1;
    b1 = temp;
  } while (load_idx < vec_dim);

  snrt_cluster_hw_barrier();

  if (cid == 0) {
    performance_timer = benchmark_get_cycle() - performance_timer;
  }


  #ifdef TIMING
  if (cid == 0) {
    printf("The DMA wait took %u cycles on average.\n", dma_wait_tot / iter);
    printf("The calculation took %u cycles on average.\n", calc_timer_avg / iter);
  }
  #endif

  if (cid == 0) {
    long unsigned int performance = 1000 * 2 * vec_dim / performance_timer;
    long unsigned int utilization =
    performance / (2 * num_cores * SNRT_NFPU_PER_CORE);

    printf("\n----- (%d) dp fdotp Simple -----\n", vec_dim);
    printf("The calculation took %u cycles.\n", performance_timer);
    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
      performance, utilization);
    }

  printf("Result of core %u: %f\n", cid, res);

  snrt_cluster_hw_barrier();

  return res;
}


int main() {
  int ret = 0;

  const unsigned int cid = snrt_cluster_core_idx();
  const unsigned int num_cores = snrt_cluster_core_num();

  const unsigned int dim = dotp_l.M;
  const unsigned int T_S = sizeof(double);

  const unsigned int NUM_BANKS = 16;

  const unsigned int SCALAR = 8;
  const unsigned int BUF_SIZE = num_cores * NUM_BANKS * SCALAR; // in doubles


  // Allocate the vector space in L1 memory
  if (cid == 0) {
    l1_buf = (double *)snrt_l1alloc(BUF_SIZE * T_S);
    result_ma = (double *)snrt_l1alloc(num_cores * T_S);
    result_simple = (double *)snrt_l1alloc(num_cores * T_S);
    snrt_memset(result_ma, 0, num_cores * T_S);
    snrt_memset(result_simple, 0, num_cores * T_S);
  }

  snrt_cluster_hw_barrier();

  result_ma[cid] = dp_dotp_db_ma(
    (double *)dotp_A_dram,
    (double *)dotp_B_dram,
    dim,
    l1_buf,
    BUF_SIZE
  );

  // Check and display results
  if (cid == 0) {
    double res_ma = result[0] + result[1];
    if (fp_check(res_ma, dotp_result)) {
      printf("\033[31;1mMA Error\033[0m: Result = %f, Golden = %f\n", res_ma, dotp_result);
      ret = -1;
    }
  }

  snrt_cluster_hw_barrier();

  // result_simple[cid] = dp_dotp_db_simple(
  //   (double *)dotp_A_dram,
  //   (double *)dotp_B_dram,
  //   dim,
  //   l1_buf,
  //   BUF_SIZE
  // );

  // if (cid == 0) {
  //   double res_simple = result[0] + result[1];
  //   if (fp_check(res_simple, dotp_result)) {
  //     printf("\033[31;1mSimple Error\033[0m: Result = %f, Golden = %f\n", res_simple, dotp_result);
  //     ret = -1;
  //   }
  // }
  // snrt_cluster_hw_barrier();

  return ret;
}

