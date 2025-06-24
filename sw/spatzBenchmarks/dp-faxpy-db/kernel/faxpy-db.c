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

// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>

#include "faxpy-db.h"

// 64-bit AXPY: y = a * x + y
void faxpy_db_v64b(const double a, const double *x, const double *y,
                unsigned int avl) {
  const unsigned int VL = 8; // Vector length
  const unsigned int LJ = 4 * VL; // Load jump size (in bytes)

  const unsigned int cid = snrt_cluster_core_idx();

  unsigned int vl;

  printf("core %u - Starting faxpy_db_v64b with a: %f, x: %p, y: %p, avl: %u\n", cid, a, x, y, avl);

  // Set the VL
  asm volatile("vsetvli %0, %1, e64, m8, ta, ma" : "=r"(vl) : "r"(VL));
  // Start the initial load
  asm volatile("vle64.v v0, (%0)" ::"r"(x));
  asm volatile("vle64.v v8, (%0)" ::"r"(y));
  printf("core %u - Loaded x: %p, y: %p\n", cid, x, y);
  x += LJ;
  y += LJ;

  // Stripmine and accumulate a partial vector
  do {
    // Set the vl
    // asm volatile("vsetvli %0, %1, e64, m8, ta, ma" : "=r"(vl) : "r"(VL));

    // Load vectors
    asm volatile("vle64.v v16, (%0)" ::"r"(x));
    asm volatile("vle64.v v24, (%0)" ::"r"(y));
    printf("core %u - Loaded x: %p, y: %p\n", cid, x, y);

    // Multiply-accumulate
    asm volatile("vfmacc.vf v8, %0, v0" ::"f"(a));
    asm volatile("vse64.v v8, (%0)" ::"r"(y-LJ));
    x += LJ;
    y += LJ;

    avl -= vl;
    if (avl <= 0)
      break;

    // Load vectors
    asm volatile ("vle64.v v0, (%0)" ::"r"(x));
    asm volatile ("vle64.v v8, (%0)" ::"r"(y));
    printf("core %u - Loaded x: %p, y: %p\n", cid, x, y);

    // Multiply-accumulate
    asm volatile("vfmacc.vf v24, %0, v0" ::"f"(a));
    asm volatile("vse64.v v24, (%0)" ::"r"(y-LJ));
    x += LJ;
    y += LJ;

    avl -= vl;
  } while (avl > 0);
}

void faxpy_v64b(const double a, const double *x, const double *y,
                unsigned int avl) {
  unsigned int vl;

  // Stripmine and accumulate a partial vector
  do {
    // Set the vl
    asm volatile("vsetvli %0, %1, e64, m8, ta, ma" : "=r"(vl) : "r"(avl));

    // Load vectors
    asm volatile("vle64.v v0, (%0)" ::"r"(x));
    asm volatile("vle64.v v8, (%0)" ::"r"(y));

    // Multiply-accumulate
    asm volatile("vfmacc.vf v8, %0, v0" ::"f"(a));

    // Store results
    asm volatile("vse64.v v8, (%0)" ::"r"(y));

    // Bump pointers
    x += vl;
    y += vl;
    avl -= vl;
  } while (avl > 0);
}

// 32-bit AXPY: y = a * x + y
void faxpy_v32b(const float a, const float *x, const float *y,
                unsigned int avl) {
  unsigned int vl;

  // Stripmine and accumulate a partial vector
  do {
    // Set the vl
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));

    // Load vectors
    asm volatile("vle32.v v0, (%0)" ::"r"(x));
    asm volatile("vle32.v v8, (%0)" ::"r"(y));

    // Multiply-accumulate
    asm volatile("vfmacc.vf v8, %0, v0" ::"f"(a));

    // Store results
    asm volatile("vse32.v v8, (%0)" ::"r"(y));

    // Bump pointers
    x += vl;
    y += vl;
    avl -= vl;
  } while (avl > 0);
}

// 16-bit AXPY: y = a * x + y
void faxpy_v16b(const _Float16 a, const _Float16 *x, const _Float16 *y,
                unsigned int avl) {
  unsigned int vl;

  // Stripmine and accumulate a partial vector
  do {
    // Set the vl
    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

    // Load vectors
    asm volatile("vle16.v v0, (%0)" ::"r"(x));
    asm volatile("vle16.v v8, (%0)" ::"r"(y));

    // Multiply-accumulate
    asm volatile("vfmacc.vf v8, %0, v0" ::"f"(a));

    // Store results
    asm volatile("vse16.v v8, (%0)" ::"r"(y));

    // Bump pointers
    x += vl;
    y += vl;
    avl -= vl;
  } while (avl > 0);
}
