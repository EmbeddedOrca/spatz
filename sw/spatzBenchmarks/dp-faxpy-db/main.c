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
// double *a;
double *x;
double *y;

static inline int fp_check(const double a, const double b) {
  const double threshold = 0.00001;

  // Absolute value
  double comp = a - b;
  if (comp < 0)
    comp = -comp;

  return comp > threshold;
}

void dp_faxpy_db_ma(
  double a, // NOTE: Shoule be in L1 memory already
  double *axpy_X_dram,
  double *axpy_Y_dram,
  unsigned int vec_dim,

  double *l1_buf,
  unsigned int l1_buf_len
) {
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  const unsigned int MEMORY_BANKS = 2;
  const unsigned int T_S = sizeof(double);
  const unsigned int CHUNK_SIZE = MEMORY_BANKS / 2; // DMA accesses half of the memory banks at a time

  // The size of the DMA transfer/operation is given by the memory size constraint size, the number of vectors, and a factor of 2 for double buffering (half of the memory can be used at a time)
  const unsigned int NUM_CHUNKS = l1_buf_len / 2 / 2 / (CHUNK_SIZE);
  double *x = l1_buf;
  double *y = l1_buf + (l1_buf_len / 2);

  unsigned int load_idx = 0;
  unsigned int calc_timer = 0;

  if (cid == 0) {
    // Start the initial DMA transfer
    snrt_dma_start_2d(
      x,
      axpy_X_dram + load_idx,
      CHUNK_SIZE * T_S,
      2 * CHUNK_SIZE * T_S,
      CHUNK_SIZE * T_S,
      NUM_CHUNKS
    );
    snrt_dma_start_2d(
      y,
      axpy_Y_dram + load_idx,
      CHUNK_SIZE * T_S,
      2 * CHUNK_SIZE * T_S,
      CHUNK_SIZE * T_S,
      NUM_CHUNKS
    );
    DEBUG(printf("Dram %p, X %p, Y %p, Num: %u\n",
      axpy_X_dram + load_idx, x,
      y, CHUNK_SIZE * NUM_CHUNKS), DBG_LVL_DBG);
    calc_timer = benchmark_get_cycle();
  }

  snrt_cluster_hw_barrier();
  unsigned int iter = 0; // The keep track of which side of the memory we are using currently

  do {
    if (cid == 0)
      snrt_dma_wait_all();

      load_idx += CHUNK_SIZE * NUM_CHUNKS; // increment the index of the next load

      if (cid == 0) {
        // Start the DMA transfer on chunk i + 1
        if (load_idx < vec_dim) {
          unsigned store_offset = ((iter + 1) % 2) * CHUNK_SIZE;
          snrt_dma_start_2d(
            x + store_offset,
            axpy_X_dram + load_idx,
            CHUNK_SIZE * T_S,
            2 * CHUNK_SIZE * T_S,
            CHUNK_SIZE * T_S,
            NUM_CHUNKS
          );
          snrt_dma_start_2d(
            y + store_offset,
            axpy_Y_dram + load_idx,
            CHUNK_SIZE * T_S,
            2 * CHUNK_SIZE * T_S,
            CHUNK_SIZE * T_S,
            NUM_CHUNKS
          );
          DEBUG(printf("Dram %p, X %p, Y %p, Num: %u\n",
            axpy_X_dram + load_idx, x + store_offset,
            y + store_offset, CHUNK_SIZE * NUM_CHUNKS), DBG_LVL_DBG);
        }
      }


      // Do the calculation on both cores
      unsigned int calc_width = CHUNK_SIZE;
      unsigned left_right = (iter % 2) * CHUNK_SIZE; // The offset to the left or right side of the memory
      unsigned core_offset = cid * MEMORY_BANKS;

      snrt_cluster_hw_barrier();

      faxpy_db_v64b(
        a,
        x + left_right + core_offset,
        y + left_right + core_offset,
        calc_width * NUM_CHUNKS / num_cores
      );

      snrt_cluster_hw_barrier();

      // Store the result of iteration i back to DRAM
      if (cid == 0) {
        snrt_dma_start_2d(
          axpy_Y_dram + load_idx - (CHUNK_SIZE * NUM_CHUNKS),
          y + left_right,
          CHUNK_SIZE * T_S,
          CHUNK_SIZE * T_S,
          2 * CHUNK_SIZE * T_S,
          NUM_CHUNKS
        );
        DEBUG(printf("Load index: %u, Iteration: %u\n", load_idx, iter), DBG_LVL_DBG);
      }

      iter++;
  } while (load_idx < vec_dim);

  // We need to wait for the final DMA transfer to finish
  if (cid == 0)
    snrt_dma_wait_all();

  snrt_cluster_hw_barrier();

  if (cid == 0)
    calc_timer = benchmark_get_cycle() - calc_timer;


  if (cid == 0) {
    long unsigned int performance = 1000 * 2 * vec_dim / calc_timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE);

    printf("\n----- (%d) MA DB axpy -----\n", vec_dim);
    printf("The execution took %u cycles.\n", calc_timer);
    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }
}

void dp_faxpy_db_simple(
  double a, // NOTE: Shoule be in L1 memory already
  double *axpy_X_dram,
  double *axpy_Y_dram,
  unsigned int vec_dim,

  double *l1_buf,
  unsigned int l1_buf_len
) {
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  const unsigned int MEMORY_BANKS = 16;
  const unsigned int T_S = sizeof(double);
  const unsigned int CHUNK_SIZE = MEMORY_BANKS / 2; // DMA accesses half of the memory banks at a time

  // The size of the DMA transfer/operation is given by the memory size constraint size, the number of vectors, and a factor of 2 for double buffering (half of the memory can be used at a time)
  const unsigned int NUM_CHUNKS = l1_buf_len / 2 / 2 / CHUNK_SIZE;
  double *x = l1_buf;
  double *y = l1_buf + (l1_buf_len / 2);
  if (cid == 0) {
    DEBUG(printf("Address of l1_buf: %p\n", l1_buf), DBG_LVL_DBG);
    DEBUG(printf("Address of x: %p, y: %p\n", x, y), DBG_LVL_DBG);
  }

  unsigned int load_idx = 0;
  unsigned int calc_timer = 0;

  if (cid == 0) {
    // Start the initial DMA transfer
    snrt_dma_start_1d(
      x,
      axpy_X_dram + load_idx,
      CHUNK_SIZE * NUM_CHUNKS * T_S
    );
    snrt_dma_start_1d(
      y,
      axpy_Y_dram + load_idx,
      CHUNK_SIZE * NUM_CHUNKS * T_S
    );
    DEBUG(printf("Dram %p, X %p, Y %p, Num: %u\n",
      axpy_X_dram + load_idx, x,
      y, CHUNK_SIZE * NUM_CHUNKS), DBG_LVL_DBG);
    calc_timer = benchmark_get_cycle();
  }
  snrt_cluster_hw_barrier();

  unsigned iter = 0;

  do {

    if (cid == 0)
      snrt_dma_wait_all();

    load_idx += CHUNK_SIZE * NUM_CHUNKS;

    if (cid == 0) {
      // Start the DMA transfer on chunk i + 1
      if (load_idx < vec_dim) {
        unsigned store_offset = ((iter + 1) % 2) * CHUNK_SIZE * NUM_CHUNKS;
        snrt_dma_start_1d(
          x + store_offset,
          axpy_X_dram + load_idx,
          CHUNK_SIZE * NUM_CHUNKS * T_S
        );
        snrt_dma_start_1d(
          y + store_offset,
          axpy_Y_dram + load_idx,
          CHUNK_SIZE * NUM_CHUNKS * T_S
        );
        DEBUG(printf("Dram %p, X %p, Y %p, Num: %u\n",
          axpy_X_dram + load_idx, x + store_offset,
          y + store_offset, CHUNK_SIZE * NUM_CHUNKS), DBG_LVL_DBG);
      }
    }

    unsigned int calc_size = CHUNK_SIZE * NUM_CHUNKS / num_cores;
    unsigned calc_offset = (iter % 2) * CHUNK_SIZE * NUM_CHUNKS + cid * calc_size;
    DEBUG(printf("Core %u, calc_offset: %u, calc_size: %u\n", cid, calc_offset, calc_size), DBG_LVL_ERR);

    faxpy_v64b(
      a,
      x + calc_offset,
      y + calc_offset,
      calc_size
    );

    snrt_cluster_hw_barrier();

    if (cid == 0) {
      snrt_dma_start_1d(
        axpy_Y_dram + load_idx - (CHUNK_SIZE * NUM_CHUNKS),
        y + calc_offset,
        CHUNK_SIZE * NUM_CHUNKS * T_S
      );
      // for (unsigned int i = 0; i < 2 * calc_size; i++) {
      //   DEBUG(printf("Y[%u] = %f\n", load_idx - (CHUNK_SIZE * NUM_CHUNKS) + i,
      //                (float)(y + calc_offset)[i]),
      //         DBG_LVL_ERR);
      // }
    }

    iter++;

  } while (load_idx < vec_dim);

  if (cid == 0)
    snrt_dma_wait_all();

  snrt_cluster_hw_barrier();

  if (cid == 0)
    calc_timer = benchmark_get_cycle() - calc_timer;

  if (cid == 0) {
    long unsigned int performance = 1000 * 2 * vec_dim / calc_timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE);

    printf("\n----- (%d) Simple DB axpy -----\n", vec_dim);
    printf("The execution took %u cycles.\n", calc_timer);
    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }
}

int main() {
  int ret = 0;

  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  const unsigned int dim = axpy_l.M;
  const unsigned int T_S = sizeof(double);

  const unsigned int SCALAR = 1;
  const unsigned int BUF_SIZE = 2 * 2 * 8 * SCALAR; // in doubles

  double *temp_buf = NULL;

  // Allocate the matrices
  if (cid == 0) {
    l1_buf = (double *)snrt_l1alloc(BUF_SIZE * T_S);

    DEBUG(printf("Address of l1_buf: %p\n", l1_buf), DBG_LVL_DBG);

    // Copy the original y to a temporary buffer
    for (unsigned int i = 0; i < dim; i++) {
      y_temp[i] = axpy_Y_dram[i];
    }
  }
  double a = axpy_alpha_dram; // The scalar value for the AXPY operation
  if (cid ==0)
    DEBUG(printf("Value of a: %f\n", (float)a), DBG_LVL_DBG);

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  dp_faxpy_db_ma(
    a, // The scalar value
    (double *)axpy_X_dram, // The input vector X
    (double *)axpy_Y_dram, // The input vector Y
    dim, // The dimension of the vectors
    l1_buf, // The L1 buffer for DMA transfers
    BUF_SIZE // The size of the L1 buffer in doubles
  );

  if (cid == 0) {
    for (unsigned int i = 0; i < dim; i++) {
      if (fp_check(axpy_Y_dram[i], axpy_GR_dram[i])) {
        printf("Error: Index %d -> Result = %f, Expected = %f\n", i,
               (float)axpy_Y_dram[i], (float)axpy_GR_dram[i]);
      } else {
        printf("Success: Index %d -> Result = %f, Expected = %f\n", i, (float)axpy_Y_dram[i], (float)axpy_GR_dram[i]);
      }
    }
  }

  snrt_cluster_hw_barrier();

  if (cid == 0) {
    // Copy the original y to a temporary buffer
    for (unsigned int i = 0; i < dim; i++) {
      axpy_Y_dram[i] = y_temp[i];
    }
  }

  snrt_cluster_hw_barrier();

  dp_faxpy_db_simple(
    a, // The scalar value
    (double *)axpy_X_dram, // The input vector X
    (double *)axpy_Y_dram, // The input vector Y
    dim, // The dimension of the vectors
    l1_buf, // The L1 buffer for DMA transfers
    BUF_SIZE // The size of the L1 buffer in doubles
  );

  if (cid == 0) {
    for (unsigned int i = 0; i < dim; i++) {
      if (fp_check(axpy_Y_dram[i], axpy_GR_dram[i])) {
        printf("Error: Index %d -> Result = %f, Expected = %f\n", i,
               (float)axpy_Y_dram[i], (float)axpy_GR_dram[i]);
      }
    }
  }

  // Wait for core 0 to finish displaying results
  snrt_cluster_hw_barrier();

  return 0;
}
