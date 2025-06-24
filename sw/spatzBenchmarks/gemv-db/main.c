// Copyright 2025 ETH Zurich and University of Bologna.
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

// Author: Navaneeth Kunhi Purayil, ETH Zurich <nkunhi@iis.ee.ethz.ch>

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include "kernel/gemv.c"
#include DATAHEADER

#if (PREC == 64)
#define T double
#elif (PREC == 32)
#define T float
#elif (PREC == 16)
#define T _Float16
#else
#define T double
#endif

T *a;
T *b;
T result[256] __attribute__((section(".data")));
T *l1_buf;

static inline int fp_check(const T a, const T b)
{
  const T threshold = 0.001;

  // Absolute value
  T comp = a - b;
  if (comp < 0)
    comp = -comp;

  return comp > threshold;
}

unsigned int gemv_db(
    double *A_dram,
    double *b_dram,
    double *x_dram,
    unsigned int M,
    unsigned int N,

    double *l1_buf,
    unsigned int l1_buf_size)
{
  unsigned int cid = snrt_cluster_core_idx();
  unsigned int num_cores = snrt_cluster_core_num();

  double *b = l1_buf;
  double *A0 = b + N;
  double *A1 = A0 + num_cores * N;
  double *x = A1 + num_cores * N;

  double *A_temp = A0;
  unsigned int calc_timer = 0;

  if (x + M > l1_buf + l1_buf_size)
  {
    printf("Error: L1 buffer size is too small for the matrices.\n");
    return -1;
  }

  if (cid == 0)
  {
    snrt_dma_start_1d(
        b,
        b_dram,
        N * sizeof(T));
    snrt_dma_start_1d(
        A0,
        A_dram,
        num_cores * N * sizeof(T));
    calc_timer = benchmark_get_cycle();
  }
  unsigned m_load = 0;
  double *x_store = x;

  do
  {
    m_load += num_cores;

    if (cid == 0 && m_load < M)
    {
      snrt_dma_wait_all();
      snrt_dma_start_1d(
          A1,
          A_dram + m_load * N,
          num_cores * N * sizeof(T));
    }

    snrt_cluster_hw_barrier();

    x_store[cid] = fdotp_v64b(
      A0 + cid * N,
      b,
      N
    );

    // printf("Core %d: x[%d] = %f\n", cid, m_load - 2 + cid, x_store[cid]);
    x_store += num_cores;

    // Swap the pointers for the next iteration
    A_temp = A0;
    A0 = A1;
    A1 = A_temp;
  } while (m_load < M);

  snrt_cluster_hw_barrier();

  if (cid == 0)
  {
    snrt_dma_start_1d(
      x_dram,
      x,
      M * sizeof(T)
    );
    snrt_dma_wait_all();
  }

  snrt_cluster_hw_barrier();

  return benchmark_get_cycle() - calc_timer;
}

int main()
{
  int ret = 0;
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  unsigned int l1_buf_len = 256 * 6;
  if (cid == 0)
  {
    l1_buf = (T *)snrt_l1alloc(l1_buf_len * sizeof(T));
  }

  snrt_cluster_hw_barrier();

  unsigned int calc_timer = gemv_db(
      gemv_A_dram,
      gemv_B_dram,
      result,

      gemv_l.M,
      gemv_l.N,

      l1_buf,
      l1_buf_len
  );

  if (cid == 0)
  {
    for (int i = 0; i < gemv_l.M; i++)
    {
      if (fp_check(result[i], gemv_result[i]))
      {
        printf("Error: ID: %i Result = %f, Golden = %f\n", i, result[i], gemv_result[i]);
        ret = -1;
      }
    }
  }
  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // Check and display results
  if (cid == 0)
  {
    long unsigned int performance = 1000 * 2 * gemv_l.M * gemv_l.N / calc_timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE * (8 / sizeof(T)));

    printf("\n----- (%d x %d) x (%d x 1) gemv -----\n", gemv_l.M, gemv_l.N, gemv_l.N);
    printf("The calculation took %u cycles.\n", calc_timer);

    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }

  // Wait for core 0 to finish displaying results
  snrt_cluster_hw_barrier();
  return ret;
}
