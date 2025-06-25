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

int check_result(double *result, unsigned int M);
void print_result(unsigned int calc_timer);

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
        N);

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
        M * sizeof(T));
    snrt_dma_wait_all();
  }

  snrt_cluster_hw_barrier();

  return benchmark_get_cycle() - calc_timer;
}

unsigned int gemv_db_multirow(
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

  const unsigned int M_PER_CORE = (l1_buf_size - M - N) / N / 4;
  printf("Number of rows: %d\n", M_PER_CORE );
  const unsigned int ELEM_PER_CORE = M_PER_CORE * N;

  const unsigned CORE_RES_OFFSET = M_PER_CORE * cid;
  const unsigned CORE_CALC_OFFSET = cid * ELEM_PER_CORE;
  const unsigned M_TOT = M_PER_CORE * num_cores;
  const unsigned NUM_ELEM_TOT = ELEM_PER_CORE * num_cores;

  double *b = l1_buf;
  double *A0 = b + N;
  double *A1 = A0 + num_cores * ELEM_PER_CORE;
  double *x = A1 + num_cores * ELEM_PER_CORE;

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
        NUM_ELEM_TOT * sizeof(T));
    snrt_dma_wait_all();
    calc_timer = benchmark_get_cycle();
  }

  // Count the number of rows loaded
  unsigned m_load = 0;

  // Pointer to store results
  double *x_store = x;

  do
  {
    m_load += M_TOT;

    if (cid == 0 && m_load < M) // && m_load < M
    {
      // snrt_dma_wait_all();
      snrt_dma_start_1d(
          A1,
          A_dram + m_load * N,
          NUM_ELEM_TOT * sizeof(T));
    }

    snrt_cluster_hw_barrier();

    // for (unsigned int i = 0; i < m; i++) {
    //   x_store[cid * M_PER_CORE + i] = fdotp_v64b(
    //     A0 + (i + cid * m) * N,
    //     b,
    //     N
    //   );
    // }

    gemv_v64b(
        A0 + CORE_CALC_OFFSET,
        b,
        x_store + CORE_RES_OFFSET,
        M_PER_CORE ,
        N
    );

    x_store += M_TOT;

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
        M * sizeof(T));
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
  unsigned int calc_timer = 0;

  unsigned int l1_buf_len = 16 * 4 * gemv_l.N + gemv_l.M + gemv_l.N;
  if (cid == 0)
  {
    l1_buf = (T *)snrt_l1alloc(l1_buf_len * sizeof(T));
  }

  snrt_cluster_hw_barrier();

  calc_timer = gemv_db(
      gemv_A_dram,
      gemv_B_dram,
      result,

      gemv_l.M,
      gemv_l.N,

      l1_buf,
      l1_buf_len);

  printf("\n----- (%d x %d) x (%d x 1) gemv_db -----\n", gemv_l.M, gemv_l.N, gemv_l.N);
  ret = check_result(result, gemv_l.M);
  print_result(calc_timer);

  // Reset the result array and l1 buffer
  if (cid == 0)
  {
    for (unsigned int i = 0; i < gemv_l.M; i++)
    {
      result[i] = 0.0;
    }
    snrt_memset(l1_buf, 0, l1_buf_len * sizeof(T));
  }

  snrt_cluster_hw_barrier();

  calc_timer = gemv_db_multirow(
      gemv_A_dram,
      gemv_B_dram,
      result,

      gemv_l.M,
      gemv_l.N,

      l1_buf,
      l1_buf_len);

  printf("\n----- (%d x %d) x (%d x 1) gemv_db_multirow -----\n", gemv_l.M, gemv_l.N, gemv_l.N);
  ret = check_result(result, gemv_l.M);
  print_result(calc_timer);

  return ret;
}

int check_result(double *result, unsigned int M)
{
  unsigned int cid = snrt_cluster_core_idx();
  unsigned int num_cores = snrt_cluster_core_num();
  int ret = 0;

  if (cid == 0)
  {
    for (int i = 0; i < M; i++)
    {
      if (fp_check(result[i], gemv_result[i]))
      {
        printf("Error: ID: %i Result = %f, Golden = %f\n", i, result[i], gemv_result[i]);
        ret = -1;
      }
    }
  }
  snrt_cluster_hw_barrier();
  return ret;
}

void print_result(unsigned int calc_timer)
{
  unsigned int cid = snrt_cluster_core_idx();
  unsigned int num_cores = snrt_cluster_core_num();
  if (cid == 0)
  {
    long unsigned int performance = 1000 * 2 * gemv_l.M * gemv_l.N / calc_timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE * (8 / sizeof(T)));

    printf("The calculation took %u cycles.\n", calc_timer);

    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }
  snrt_cluster_hw_barrier();
}
