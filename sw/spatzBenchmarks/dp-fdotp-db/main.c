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

double *l1_buf;
double *a;
double *b;
double *result;
double dotp_res = 0.0;

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
  unsigned int l1_buf_len, // In sizeof doubles

  double *result
) {
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  // HW and algo configuration
  const unsigned int MEMORY_BANKS = 16;
  const unsigned int T_S = sizeof(double);
  const unsigned int CHUNK_SIZE = MEMORY_BANKS / 2; // DMA blocks half of the memory banks

  // The number of chunks we can load at a time is given by the useable memory size
  // devided by the number of vectors and a factor of 2 for the double buffering
  // and the size of a single chunk
  const unsigned int NUM_CHUNKS = l1_buf_len / 2 / 2 / (CHUNK_SIZE);
  double *a = l1_buf; // First half for the first vector
  double *b = l1_buf + (l1_buf_len / 2); // Second half for the second vector

  // Reset timer
  unsigned int load_timer = (unsigned int)-1;
  unsigned int calc_timer = (unsigned int)-1;

  unsigned int load_idx = 0;

  unsigned num_iter = vec_dim / (CHUNK_SIZE * NUM_CHUNKS);
  if (cid == 0) {
    DEBUG(printf("Expected iterations: %u\n", num_iter), DBG_LVL_DBG);
  }

  // Start the initial DMA transfer
  if (cid == 0) {
    snrt_dma_start_2d(
      a,
      dotp_A_dram + load_idx,
      CHUNK_SIZE * T_S,
      2 * CHUNK_SIZE * T_S,
      CHUNK_SIZE * T_S,
      NUM_CHUNKS
    );
    snrt_dma_start_2d(
      b,
      dotp_B_dram + load_idx,
      CHUNK_SIZE * T_S,
      2 * CHUNK_SIZE * T_S,
      CHUNK_SIZE * T_S,
      NUM_CHUNKS
    );
    calc_timer = benchmark_get_cycle();
  }

  snrt_cluster_hw_barrier();

  if (cid == 0)
    start_kernel();

  unsigned iter = 0;

  do {

    // Wait for the data load of the current calc data
    if (cid == 0)
      snrt_dma_wait_all();

    load_idx += CHUNK_SIZE * NUM_CHUNKS;

    snrt_cluster_hw_barrier();

    // Start the DMA transfer on chunk i + 1
    if (cid == 0) {
      if (load_idx < vec_dim) {
        unsigned store_offset = ((iter + 1) % 2) * CHUNK_SIZE;

        snrt_dma_start_2d(
          a + store_offset,
          dotp_A_dram + load_idx,
          T_S * CHUNK_SIZE,
          2 * CHUNK_SIZE * T_S,
          CHUNK_SIZE * T_S,
          NUM_CHUNKS
        );
        snrt_dma_start_2d(
          b + store_offset,
          dotp_B_dram + load_idx,
          CHUNK_SIZE * T_S,
          2 * CHUNK_SIZE * T_S,
          CHUNK_SIZE * T_S,
          NUM_CHUNKS
        );
      }
    }

    // Save the result of iteration i
    // unsigned calc_width = CHUNCK_SIZE >> 1;
    // unsigned calc_offset = (iter % 2) * CHUNCK_SIZE + calc_width * cid;
    unsigned calc_width = CHUNK_SIZE;
    unsigned left_right = (iter % 2) * CHUNK_SIZE;
    unsigned core_offset = cid * MEMORY_BANKS;

    unsigned calc_offset = left_right + core_offset;
    result[cid] = fdotp_v64b_ma(
      a + calc_offset,
      b + calc_offset,
      calc_width * NUM_CHUNKS / num_cores,
      result[cid]
    );

    iter++;
  } while (load_idx < vec_dim);

  snrt_cluster_hw_barrier();

  // printf("Result of core %u: %f\n", cid, result[cid]);

  if (cid == 0) {
    calc_timer = benchmark_get_cycle() - calc_timer;
  }

  snrt_cluster_hw_barrier();

  // End dump, Record the time
  if (cid == 0) {
    stop_kernel();

    long unsigned int performance = 1000 * 2 * dotp_l.M / calc_timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE);

    printf("\n----- (%d) dp fdotp MA -----\n", dotp_l.M);
    printf("The calculation took %u cycles.\n", calc_timer);
    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }

  snrt_cluster_hw_barrier();

  // Accumulate the result into res
  double res = 0.0;
  if (cid == 0) {
    // DEBUG(printf("Accumulate: %f, %f\n", result[0], result[1]), DBG_LVL_ERR);
    res = vreduce_sum(result, num_cores);
    DEBUG(printf("Final result: %f\n", res), DBG_LVL_ERR);
  }

  snrt_cluster_hw_barrier();

  return res;
}

/**
 * @brief Implementation of the dot product using simple double buffering without being aware of the L1 memory layout.
 *
 * This function computes the dot product of two vectors A and B of size `dim`
 */
double dp_dotp_db_simple(
  double *dotp_A_dram,
  double *dotp_B_dram,
  unsigned int vec_dim,

  double *l1_buf,
  unsigned int l1_buf_len, // In sizeof doubles

  double *result
) {
  const unsigned int cid = snrt_cluster_core_idx();
  const unsigned int num_cores = snrt_cluster_core_num();

  result[cid] = 0.0;

  const unsigned int MEMORY_BANKS = 16;
  const unsigned int T_S = sizeof(double);
  const unsigned int CHUNCK_SIZE = MEMORY_BANKS / 2;

  const unsigned int NUM_CHUNCKS = l1_buf_len / 2 / 2 / CHUNCK_SIZE;
  double *a = l1_buf; // First half for the first vector
  double *b = l1_buf + (l1_buf_len / 2); // Second half for the second vector

  // Reset timer
  unsigned int load_timer = (unsigned int)-1;
  unsigned int calc_timer = (unsigned int)-1;

  unsigned int load_idx = 0;

  unsigned num_iter = vec_dim / (CHUNCK_SIZE * NUM_CHUNCKS);
  if (cid == 0) {
    DEBUG(printf("Expected iterations: %u\n", num_iter), DBG_LVL_DBG);
  }

  if (cid == 0) {
    snrt_dma_start_1d(
      a,
      dotp_A_dram + load_idx, // load index is in size of type
      CHUNCK_SIZE * NUM_CHUNCKS * T_S // in bytes
    );
    snrt_dma_start_1d(
      b,
      dotp_B_dram + load_idx, // load index is in size of type
      CHUNCK_SIZE * NUM_CHUNCKS * T_S // in bytes
    );
    DEBUG(printf("Dram %p, A %p, B %p, Num: %u\n", dotp_A_dram + load_idx, a, b, CHUNCK_SIZE * NUM_CHUNCKS), DBG_LVL_DBG);
    calc_timer = benchmark_get_cycle();
  }

  snrt_cluster_hw_barrier();

  if (cid == 0)
    start_kernel();

  unsigned iter = 0;

  do {
    // Wait for the data load of the current calc data
    if (cid == 0)
      snrt_dma_wait_all();

    load_idx += CHUNCK_SIZE * NUM_CHUNCKS; // increment the index of the next load

    snrt_cluster_hw_barrier();

    // Start the DMA transfer on chunk i + 1
    if (cid == 0) {
      if (load_idx < vec_dim) {
        unsigned store_offset = CHUNCK_SIZE * NUM_CHUNCKS * ((iter + 1) % 2);

        snrt_dma_start_1d(
          a + store_offset,
          dotp_A_dram + load_idx, // load index is in size of type
          CHUNCK_SIZE * NUM_CHUNCKS * T_S // in bytes
        );
        snrt_dma_start_1d(
          b + store_offset,
          dotp_B_dram + load_idx, // load index is in size of type
          CHUNCK_SIZE * NUM_CHUNCKS * T_S // in bytes
        );
        DEBUG(printf("Dram %p, A %p, B %p, Num: %u\n", dotp_A_dram + load_idx, a + store_offset, b + store_offset, CHUNCK_SIZE * NUM_CHUNCKS), DBG_LVL_DBG);
      }
    }

    // Save the result of iteration i of each core
    unsigned calc_size = CHUNCK_SIZE * NUM_CHUNCKS / num_cores;
    unsigned calc_offset = (iter % 2) * CHUNCK_SIZE * NUM_CHUNCKS + cid * calc_size;

    result[cid] = fdotp_v64b(
      a + calc_offset,
      b + calc_offset,
      calc_size,
      result[cid]
    );

    iter++;
  } while (load_idx < vec_dim);

  snrt_cluster_hw_barrier();

  if (cid == 0) {
    calc_timer = benchmark_get_cycle() - calc_timer;
  }

  printf("Result of core %u: %f\n", cid, result[cid]);

  snrt_cluster_hw_barrier();

  // End dump, Record the time
  if (cid == 0) {
    stop_kernel();

    long unsigned int performance = 1000 * 2 * vec_dim / calc_timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE);

    printf("\n----- (%d) dp fdotp Simple -----\n", vec_dim);
    printf("The calculation took %u cycles.\n", calc_timer);
    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }

  snrt_cluster_hw_barrier();

  double res = 0.0;
  // Accumulate the result into res
  if (cid == 0) {
    DEBUG(printf("Accumulate: %f, %f\n", result[0], result[1]), DBG_LVL_ERR);
    res = vreduce_sum(result, num_cores);
    DEBUG(printf("Final result: %f\n", res), DBG_LVL_ERR);
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  return res;
}

int main() {
  int ret = 0;

  const unsigned int cid = snrt_cluster_core_idx();
  const unsigned int num_cores = snrt_cluster_core_num();

  const unsigned int dim = dotp_l.M;
  const unsigned int T_S = sizeof(double);

  const unsigned int SCALAR = 8;
  const unsigned int BUF_SIZE = 2 * 2 * 8 * SCALAR; // in doubles


  // Allocate the vector space in L1 memory
  if (cid == 0) {
    l1_buf = (double *)snrt_l1alloc(BUF_SIZE * T_S);
    result = (double *)snrt_l1alloc(num_cores * T_S);
    snrt_memset(result, 0, num_cores * T_S * 2);

    DEBUG(printf("Address of l1_buf: %p\n", l1_buf), DBG_LVL_DBG);
    DEBUG(printf("Address of result: %p\n", result), DBG_LVL_DBG);
  }

  for (unsigned i = 0; i < 2; i++) {

  snrt_cluster_hw_barrier();

  double res_ma = dp_dotp_db_ma(
    (double *)dotp_A_dram,
    (double *)dotp_B_dram,
    dim,
    l1_buf,
    BUF_SIZE,
    result
  );

  // Check and display results
  if (cid == 0) {
    if (fp_check(res_ma, dotp_result)) {
      printf("\033[31;1mMA Error\033[0m: Result = %f, Golden = %f\n", res_ma, dotp_result);
      ret = -1;
    } else {
      printf("\033[32;1mMA Success\033[0m: Result = %f, Golden = %f\n", res_ma, dotp_result);
    }
  }

  snrt_cluster_hw_barrier();

  // Reset the memory
  if (cid == 0) {
    snrt_memset(l1_buf, 0, BUF_SIZE * T_S);
    snrt_memset(result, 0, num_cores * T_S * 2);
  }

  if (cid == 0) {
    l1_buf = (double *)snrt_l1alloc(BUF_SIZE * T_S);
    result = (double *)snrt_l1alloc(num_cores * T_S);
    snrt_memset(result, 0, num_cores * T_S * 2);

    DEBUG(printf("Address of l1_buf: %p\n", l1_buf), DBG_LVL_DBG);
    DEBUG(printf("Address of result: %p\n", result), DBG_LVL_DBG);
  }

  snrt_cluster_hw_barrier();

  double res_simple = dp_dotp_db_simple(
    (double *)dotp_A_dram,
    (double *)dotp_B_dram,
    dim,
    l1_buf,
    BUF_SIZE,
    result + 2
  );

  if (cid == 0) {
    if (fp_check(res_simple, dotp_result)) {
      printf("\033[31;1mSimple Error\033[0m: Result = %f, Golden = %f\n", res_simple, dotp_result);
      ret = -1;
    } else {
      printf("\033[32;1mSimple Success\033[0m: Result = %f, Golden = %f\n", res_simple, dotp_result);
    }
  }
  snrt_cluster_hw_barrier();

  printf("\n");

}

  return ret;
}
