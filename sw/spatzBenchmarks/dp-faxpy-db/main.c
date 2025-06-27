// Copyright 2023 ETH Zurich and University of Bologna.
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

// Author: Matheus Cavalcante, ETH Zürich

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/faxpy-db.c"

// #define TIMING 1

// Logging Macros
#define DBG_LVL_NONE 0
#define DBG_LVL_ERR 1
#define DBG_LVL_WARN 2
#define DBG_LVL_INFO 3
#define DBG_LVL_DBG 4

#define DBG_LVL DBG_LVL_ERR

#define DEBUG(func, lvl) \
  if (lvl <= DBG_LVL)    \
  {                      \
    func;                \
  }

// Shared variables between the cores
double *l1_buf;
double *a;
double *x;
double *y;

// Used to store the original y vector for testing
static double y_temp[4096] __attribute__((section(".data"))) = {0};

static inline int fp_check(const double a, const double b)
{
  const double threshold = 0.00001;

  // Absolute value
  double comp = a - b;
  if (comp < 0)
    comp = -comp;

  return comp > threshold;
}

/**
 * @brief Performs a double precision AXPY operation using double buffering and memory alignment.
 *        It uses the left and right 8 banks of the L1 memory successively.
 *
 * @param a The scalar multiplier
 * @param axpy_X_dram Pointer to the X vector in DRAM
 * @param axpy_Y_dram Pointer to the Y vector in DRAM
 * @param vec_dim The number of elements in the vectors
 * @param l1_buf Pointer to the L1 buffer for double buffering
 * @param l1_buf_len The length of the L1 buffer in doubles
 */
void dp_faxpy_db_ma(
    double a,
    double *axpy_X_dram,
    double *axpy_Y_dram,
    unsigned int vec_dim,

    double *l1_buf,
    unsigned int l1_buf_len)
{
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  const unsigned int MEMORY_BANKS = 16;
  const unsigned int T_S = sizeof(double);
  const unsigned int CHUNK_SIZE = MEMORY_BANKS / 2; // DMA accesses half of the memory banks at a time

  // The size of the DMA transfer/operation is given by the memory size constraint size, the number of
  // vectors, and a factor of 2 for double buffering (half of the memory can be used at a time)
  const unsigned int NUM_CHUNKS = l1_buf_len / 2 / 2 / (CHUNK_SIZE);
  double *x = l1_buf;
  double *y = l1_buf + (l1_buf_len / 2);

  unsigned int load_idx = 0;
  unsigned int calc_timer = 0;
  unsigned int calc_timer_avg = 0;
  unsigned dma_wait_tot = 0;
  unsigned int performance_timer = 0;

  // Start the initial DMA transfer
  if (cid == 0)
  {
    snrt_dma_start_2d(
        x,
        axpy_X_dram + load_idx,
        CHUNK_SIZE * T_S,
        2 * CHUNK_SIZE * T_S,
        CHUNK_SIZE * T_S,
        NUM_CHUNKS);
    snrt_dma_start_2d(
        y,
        axpy_Y_dram + load_idx,
        CHUNK_SIZE * T_S,
        2 * CHUNK_SIZE * T_S,
        CHUNK_SIZE * T_S,
        NUM_CHUNKS);
    DEBUG(printf("Dram %p, X %p, Y %p, Num: %u\n",
                 axpy_X_dram + load_idx, x,
                 y, CHUNK_SIZE * NUM_CHUNKS),
          DBG_LVL_DBG);
    performance_timer = benchmark_get_cycle();
  }

  snrt_cluster_hw_barrier();
  unsigned int iter = 0; // The keep track of which side of the memory we are using currently

  do
  {
    if (cid == 0)
    {
#ifdef TIMING
      unsigned dma_wait = benchmark_get_cycle();
#endif
      snrt_dma_wait_all();
#ifdef TIMING
      dma_wait = benchmark_get_cycle() - dma_wait;
      dma_wait_tot += dma_wait;
#endif
    }

    load_idx += CHUNK_SIZE * NUM_CHUNKS; // increment the index of the next load

    // Start the DMA transfer on chunk i + 1
    if (cid == 0)
    {
      start_kernel();
      if (load_idx < vec_dim)
      {
        unsigned store_offset = ((iter + 1) % 2) * CHUNK_SIZE;
        snrt_dma_start_2d(
            x + store_offset,
            axpy_X_dram + load_idx,
            CHUNK_SIZE * T_S,
            2 * CHUNK_SIZE * T_S,
            CHUNK_SIZE * T_S,
            NUM_CHUNKS);
        snrt_dma_start_2d(
            y + store_offset,
            axpy_Y_dram + load_idx,
            CHUNK_SIZE * T_S,
            2 * CHUNK_SIZE * T_S,
            CHUNK_SIZE * T_S,
            NUM_CHUNKS);
        DEBUG(printf("Dram %p, X %p, Y %p, Num: %u\n",
                     axpy_X_dram + load_idx, x + store_offset,
                     y + store_offset, CHUNK_SIZE * NUM_CHUNKS),
              DBG_LVL_DBG);
      }
    }

    // Do the calculation on both cores
    unsigned int calc_width = CHUNK_SIZE;
    unsigned core_offset = cid * MEMORY_BANKS;
    unsigned left_right = (iter % 2) * CHUNK_SIZE; // The offset to the left or right side of the memory

    snrt_cluster_hw_barrier();

#ifdef TIMING
    if (cid == 0)
    {
      calc_timer = benchmark_get_cycle();
    }
#endif

    faxpy_db_v64b(
        a,
        x + left_right + core_offset,
        y + left_right + core_offset,
        calc_width * NUM_CHUNKS / num_cores);

#ifdef TIMING
    if (cid == 0)
    {
      calc_timer = benchmark_get_cycle() - calc_timer;
      calc_timer_avg += calc_timer;
    }
#endif

    snrt_cluster_hw_barrier();

    // Store the result of iteration i back to DRAM
    if (cid == 0)
    {
      start_kernel();
      snrt_dma_start_2d(
          axpy_Y_dram + load_idx - (CHUNK_SIZE * NUM_CHUNKS),
          y + left_right,
          CHUNK_SIZE * T_S,
          CHUNK_SIZE * T_S,
          2 * CHUNK_SIZE * T_S,
          NUM_CHUNKS);
      stop_kernel();
      DEBUG(printf("Load index: %u, Iteration: %u\n", load_idx, iter), DBG_LVL_DBG);
    }

    iter++;
  } while (load_idx < vec_dim);

  // We need to wait for the final DMA transfer to finish
  if (cid == 0)
    snrt_dma_wait_all();

  snrt_cluster_hw_barrier();

  if (cid == 0)
    performance_timer = benchmark_get_cycle() - performance_timer;

#ifdef TIMING
  if (cid == 0)
  {
    printf("The DMA wait took %u cycles on average.\n", dma_wait_tot / iter);
    printf("The calculation took %u cycles on average.\n", calc_timer_avg / iter);
  }
#endif

  if (cid == 0)
  {
    long unsigned int performance = 1000 * 2 * vec_dim / performance_timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE);

    printf("\n----- (%d) MA DB axpy -----\n", vec_dim);
    printf("The execution took %u cycles.\n", performance_timer);
    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }
}

/**
 * @brief Performs a double precision AXPY operation using double buffering.
 *
 * @param a The scalar multiplier
 * @param axpy_X_dram Pointer to the X vector in DRAM
 * @param axpy_Y_dram Pointer to the Y vector in DRAM
 * @param vec_dim The number of elements in the vectors
 * @param l1_buf Pointer to the L1 buffer for double buffering
 * @param l1_buf_len The length of the L1 buffer in doubles
 */
void dp_faxpy_db_simple(
    double a,
    double *axpy_X_dram,
    double *axpy_Y_dram,
    unsigned int vec_dim,

    double *l1_buf,
    unsigned int l1_buf_len)
{
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  const unsigned int MEMORY_BANKS = 16;
  const unsigned int T_S = sizeof(double);
  const unsigned int CHUNK_SIZE = MEMORY_BANKS / 2; // DMA accesses half of the memory banks at a time

  // The size of the DMA transfer/operation is given by the memory size constraint size, the number of vectors, and a factor of 2 for double buffering (half of the memory can be used at a time)
  const unsigned int NUM_CHUNKS = l1_buf_len / 2 / 2 / CHUNK_SIZE;
  double *x = l1_buf;
  double *y = l1_buf + (l1_buf_len / 2);

  unsigned int load_idx = 0;

  unsigned int calc_timer = 0;
  unsigned int calc_timer_avg = 0;
  unsigned dma_wait_tot = 0;
  unsigned int performance_timer = 0;

  if (cid == 0)
  {
    // Start the initial DMA transfer
    snrt_dma_start_1d(
        x,
        axpy_X_dram + load_idx,
        CHUNK_SIZE * NUM_CHUNKS * T_S);
    snrt_dma_start_1d(
        y,
        axpy_Y_dram + load_idx,
        CHUNK_SIZE * NUM_CHUNKS * T_S);
    DEBUG(printf("Dram %p, X %p, Y %p, Num: %u\n",
                 axpy_X_dram + load_idx, x,
                 y, CHUNK_SIZE * NUM_CHUNKS),
          DBG_LVL_DBG);
    performance_timer = benchmark_get_cycle();
  }

  unsigned iter = 0;
  snrt_dma_txid_t load_transfer = 0;

  do
  {

    if (cid == 0)
    {
#ifdef TIMING
      unsigned dma_wait = benchmark_get_cycle();
#endif
      snrt_dma_wait(load_transfer);
#ifdef TIMING
      dma_wait = benchmark_get_cycle() - dma_wait;
      dma_wait_tot += dma_wait;
#endif
    }

    load_idx += CHUNK_SIZE * NUM_CHUNKS;

    if (cid == 0)
    {
      // Start the DMA transfer on chunk i + 1
      if (load_idx < vec_dim)
      {
        unsigned store_offset = ((iter + 1) % 2) * CHUNK_SIZE * NUM_CHUNKS;
        snrt_dma_start_1d(
            x + store_offset,
            axpy_X_dram + load_idx,
            CHUNK_SIZE * NUM_CHUNKS * T_S);
        load_transfer = snrt_dma_start_1d(
            y + store_offset,
            axpy_Y_dram + load_idx,
            CHUNK_SIZE * NUM_CHUNKS * T_S);
        DEBUG(printf("Dram %p, X %p, Y %p, Num: %u\n",
                     axpy_X_dram + load_idx, x + store_offset,
                     y + store_offset, CHUNK_SIZE * NUM_CHUNKS),
              DBG_LVL_DBG);
      }
    }

    unsigned int calc_size = CHUNK_SIZE * NUM_CHUNKS / num_cores;
    unsigned calc_offset = (iter % 2) * CHUNK_SIZE * NUM_CHUNKS + cid * calc_size;

#ifdef TIMING
    if (cid == 0)
    {
      calc_timer = benchmark_get_cycle();
    }
#endif

    faxpy_v64b(
        a,
        x + calc_offset,
        y + calc_offset,
        calc_size);

#ifdef TIMING
    if (cid == 0)
    {
      calc_timer = benchmark_get_cycle() - calc_timer;
      calc_timer_avg += calc_timer;
    }
#endif

    snrt_cluster_hw_barrier();

    if (cid == 0)
    {
      snrt_dma_start_1d(
          axpy_Y_dram + load_idx - (CHUNK_SIZE * NUM_CHUNKS),
          y + calc_offset,
          CHUNK_SIZE * NUM_CHUNKS * T_S);
    }

    iter++;

  } while (load_idx < vec_dim);

  if (cid == 0)
    snrt_dma_wait_all();

  snrt_cluster_hw_barrier();

  if (cid == 0)
    performance_timer = benchmark_get_cycle() - performance_timer;

#ifdef TIMING
  if (cid == 0)
  {
    printf("The DMA wait took %u cycles on average.\n", dma_wait_tot / iter);
    printf("The calculation took %u cycles on average.\n", calc_timer_avg / iter);
  }
#endif

  if (cid == 0)
  {
    long unsigned int performance = 1000 * 2 * vec_dim / performance_timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE);

    printf("\n----- (%d) Simple DB axpy -----\n", vec_dim);
    printf("The execution took %u cycles.\n", performance_timer);
    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }
}

int main()
{
  int ret = 0;

  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  const unsigned int NUM_BANKS = 16; // Number of memory banks
  const unsigned int dim = axpy_l.M;
  const unsigned int T_S = sizeof(double);

  // The size of the L1 buffer in doubles, at least 32 doubles (16 banks * 2 cores)
  const unsigned int SCALAR = 32;
  const unsigned int BUF_SIZE = 2 * NUM_BANKS * SCALAR; // in doubles

  double *temp_buf = NULL;

  // Allocate the matrices
  if (cid == 0)
  {
    l1_buf = (double *)snrt_l1alloc(BUF_SIZE * T_S);

    // Copy the original y to a temporary buffer
    for (unsigned int i = 0; i < dim; i++)
    {
      y_temp[i] = axpy_Y_dram[i];
    }
  }
  double a = axpy_alpha_dram;

  snrt_cluster_hw_barrier();

  /// Test the first implementation
  dp_faxpy_db_ma(
      a,
      (double *)axpy_X_dram,
      (double *)axpy_Y_dram,
      dim,
      l1_buf,
      BUF_SIZE);

  if (cid == 0)
  {
    for (unsigned int i = 0; i < dim; i++)
    {
      if (fp_check(axpy_Y_dram[i], axpy_GR_dram[i]))
      {
        printf("Error: Index %d -> Result = %f, Expected = %f\n", i,
               (float)axpy_Y_dram[i], (float)axpy_GR_dram[i]);
        ret = -1;
      }
    }
  }

  snrt_cluster_hw_barrier();

  // Reset the y vector to the original values
  if (cid == 0)
  {
    for (unsigned int i = 0; i < dim; i++)
    {
      axpy_Y_dram[i] = y_temp[i];
    }
  }

  snrt_cluster_hw_barrier();

  // Test the second implementation
  dp_faxpy_db_simple(
      a,
      (double *)axpy_X_dram,
      (double *)axpy_Y_dram,
      dim,
      l1_buf,
      BUF_SIZE);

  // Check the results
  if (cid == 0)
  {
    for (unsigned int i = 0; i < dim; i++)
    {
      if (fp_check(axpy_Y_dram[i], axpy_GR_dram[i]))
      {
        printf("Error: Index %d -> Result = %f, Expected = %f\n", i,
               (float)axpy_Y_dram[i], (float)axpy_GR_dram[i]);
        ret = -1;
      }
    }
  }

  snrt_cluster_hw_barrier();

  return ret;
}
