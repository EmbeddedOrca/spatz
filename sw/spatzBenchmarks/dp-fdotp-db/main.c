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

double *a;
double *b;
double *result;
double dotp_res = 0.0;

#define MEMORY_BANKS 16
#define T_S sizeof(double)
#define CHUNCK_SIZE 8 // We always access at 512 bit at a time

#define NUM_CHUNCKS 1
#define TRANSFER_SIZE (NUM_CHUNCKS * CHUNCK_SIZE)
// #define NUM_ROW CHUNCK_SIZE / ACCESS_WIDTH

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

int main() {
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  // Reset timer
  unsigned int load_timer = (unsigned int)-1;
  unsigned int calc_timer = (unsigned int)-1;

  unsigned int load_idx = 0;
  unsigned int dim = dotp_l.M;

  unsigned num_iter = dim / (CHUNCK_SIZE * NUM_CHUNCKS);
  if (cid == 0) {
    DEBUG(printf("Expected iterations: %u\n", num_iter), DBG_LVL_DBG);
  }

  // Allocate the matrices
  if (cid == 0) {
    a = (double *)snrt_l1alloc(dim * T_S);
    b = (double *)snrt_l1alloc(dim * T_S);
    result = (double *)snrt_l1alloc(num_cores * T_S * num_iter);
    snrt_memset(result, 0, num_cores * T_S * num_iter);

    DEBUG(printf("Address of a: %p\n", a), DBG_LVL_ERR);
    DEBUG(printf("Address of b: %p\n", b), DBG_LVL_ERR);
    DEBUG(printf("Address of result: %p\n", result), DBG_LVL_ERR);
  }

  if (cid == 0) {
    calc_timer = benchmark_get_cycle();
    snrt_dma_start_2d(
      a,
      dotp_A_dram + load_idx,
      T_S * CHUNCK_SIZE,
      2 * CHUNCK_SIZE * T_S,
      CHUNCK_SIZE * T_S,
      NUM_CHUNCKS
    );
    snrt_dma_start_2d(
      b,
      dotp_B_dram + load_idx,
      CHUNCK_SIZE * T_S,
      2 * CHUNCK_SIZE * T_S,
      CHUNCK_SIZE * T_S,
      NUM_CHUNCKS
    );
  }

  snrt_cluster_hw_barrier();

  if (cid == 0)
  start_kernel();

  unsigned iter = 0;

  do {


    if (cid == 0)
      snrt_dma_wait_all();

    load_idx += CHUNCK_SIZE * NUM_CHUNCKS;

    snrt_cluster_hw_barrier();

    if (cid == 0) {
      if (load_idx < dim) {
        unsigned store_offset = ((iter + 1) % 2) * CHUNCK_SIZE;

        snrt_dma_start_2d(
          a + store_offset,
          dotp_A_dram + load_idx,
          T_S * CHUNCK_SIZE,
          2 * CHUNCK_SIZE * T_S,
          CHUNCK_SIZE * T_S,
          NUM_CHUNCKS
        );
        snrt_dma_start_2d(
          b + store_offset,
          dotp_B_dram + load_idx,
          CHUNCK_SIZE * T_S,
          2 * CHUNCK_SIZE * T_S,
          CHUNCK_SIZE * T_S,
          NUM_CHUNCKS
        );
      }
    }

    unsigned calc_width = CHUNCK_SIZE >> 1;
    unsigned calc_offset = (iter % 2) * CHUNCK_SIZE + calc_width * cid;
    // DEBUG(printf("Core %u: A: %p, B: %p, calc_width: %u \n", cid,
                //  a + calc_offset, b + calc_offset, calc_width), DBG_LVL_DBG);

    result[(iter << 1) + cid] = fdotp_v64b(
      a + calc_offset,
      b + calc_offset,
      calc_width
    );
    // result[cid] += fdotp(a + calc_offset, b + calc_offset, CHUNCK_SIZE >> 1);
    // DEBUG(printf("Core %u: %f\n", cid, result[cid]), DBG_LVL_DBG);

      iter++;
  } while (load_idx < dim);

  if (cid == 0)
    DEBUG(printf("Accumulate\n"), DBG_LVL_INFO);
  snrt_cluster_hw_barrier();

  double acc = 0.0;
  if (cid == 0) {
    acc = vreduce_sum(result, num_cores * iter);
    dotp_res = acc;
    DEBUG(printf("Final result: %f\n", dotp_res), DBG_LVL_DBG);
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // End dump, Record the time
  if (cid == 0)
    stop_kernel();
    calc_timer = benchmark_get_cycle() - calc_timer;

  if (cid == 0) {
    long unsigned int performance = 1000 * 2 * dotp_l.M / calc_timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE);

    printf("\n----- (%d) dp fdotp -----\n", dotp_l.M);
    printf("The calculation took %u cycles.\n", calc_timer);
    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }


  if (cid == 0)
    if (fp_check(dotp_res, dotp_result)) {
      printf("Error: Result = %f, Golden = %f\n", dotp_res, dotp_result);
      return -1;
    }

  // Wait for core 0 to finish displaying results
  snrt_cluster_hw_barrier();

  return 0;

}
