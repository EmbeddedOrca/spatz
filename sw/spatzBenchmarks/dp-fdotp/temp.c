int main() {
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  // Reset timer
  unsigned int load_timer = (unsigned int)-1;
  unsigned int calc_timer = (unsigned int)-1;

  unsigned int load_idx = 0;
  unsigned int calc_idx = 0;
  unsigned int end_idx = dotp_l.M;

  unsigned int load_size = 0;
  unsigned int repetition = 0;

  // Allocate the matrices
  if (cid == 0) {
    a = (double *)snrt_l1alloc(dotp_l.M * T_S);
    b = (double *)snrt_l1alloc(dotp_l.M * T_S);
    result = (double *)snrt_l1alloc(num_cores * T_S);
    snrt_memset(result, 0, num_cores * T_S);

    DEBUG(printf("Address of a: %p\n", a), DBG_LVL_ERR);
    DEBUG(printf("Address of b: %p\n", b), DBG_LVL_ERR);
    DEBUG(printf("Address of result: %p\n", result), DBG_LVL_ERR);
  }

  if (cid == 0) {
    load_timer = benchmark_get_cycle();

    get_load_size_repetition(load_idx, end_idx, &load_size, &repetition);
    DEBUG(printf("Index: %u, End Index: %u, Load size: %u, Repetition: %u\n", load_idx, end_idx, load_size, repetition), DBG_LVL_INFO);
    snrt_dma_start_2d(
      a,                //
      dotp_A_dram,       //
      load_size,       //
      2 * CHUNCK_SIZE, //
      load_size,       //
      repetition        //
    );
    snrt_dma_start_2d(
      b,                // dest
      dotp_B_dram,       // src
      load_size,     // size per repetition, in B
      2 * CHUNCK_SIZE, // dest_stride, we want to skip 8 banks
      load_size,       // src_stride, we want to load continously
      repetition       // number of rows to load at a time
    );
    DEBUG(printf("Src: %p, Dest: %p, Size: %u, Src Stride: %u, Dest Stride: %u, Repetition: %u \n", dotp_B_dram + (load_idx >> 3), b + ((load_idx >> 3) % MEMORY_BANKS), load_size, load_size, 2 * CHUNCK_SIZE, repetition), DBG_LVL_INFO);
  }
  load_idx += load_size * repetition;
  snrt_cluster_hw_barrier();

  if (cid == 0)
    start_kernel();
    calc_timer = benchmark_get_cycle();

  unsigned initial_load = 1;

  do {

    if (cid == 0) {
      snrt_dma_wait_all();
      if (initial_load) {
        load_timer = benchmark_get_cycle() - load_timer;
        initial_load = 0;
      }
    }

    snrt_cluster_hw_barrier();

    get_load_size_repetition(load_idx, end_idx, &load_size, &repetition);
    if (cid == 0) {
      if (load_idx < end_idx) {
        DEBUG(printf("Index: %u, End Index: %u, Load size: %u, Repetition: %u\n", load_idx, end_idx, load_size, repetition), DBG_LVL_INFO);
        snrt_dma_start_2d(
          a + ((load_idx >> 3) % MEMORY_BANKS),                //
          dotp_A_dram + (load_idx >> 3),       //
          load_size,       //
          2 * CHUNCK_SIZE, //
          load_size,       //
          repetition        //
        );
        snrt_dma_start_2d(
          b + ((load_idx >> 3) % MEMORY_BANKS),                // dest
          dotp_B_dram + (load_idx >> 3),       // src
          load_size,     // size per repetition, in B
          2 * CHUNCK_SIZE, // dest_stride, we want to skip 8 banks
          load_size,       // src_stride, we want to load continously
          repetition       // number of rows to load at a time
        );
        DEBUG(printf("Src: %p, Dest: %p, Size: %u, Src Stride: %u, Dest Stride: %u, Repetition: %u \n", dotp_B_dram + (load_idx >> 3), b + ((load_idx >> 3) % MEMORY_BANKS), load_size, load_size, 2 * CHUNCK_SIZE, repetition), DBG_LVL_INFO);
      }
    }

    load_idx += load_size * repetition;

    snrt_cluster_hw_barrier();

    // TODO: ADJUST KERNEL
    unsigned int calc_size = 0;
    // get_calc_offset_size(cid, num_cores, &calc_idx, &calc_size);
    // DEBUG(printf("Core %u: calc_idx = %u, calc_size = %u\n", cid, calc_idx, calc_size), DBG_LVL_INFO);


    calc_idx = calc_idx % MEMORY_WIDTH + cid * (MEMORY_BANKS / num_cores);
    calc_size = 4;
    result[cid] += fdotp_v64b(a + (calc_idx >> 3), b + (calc_idx >> 3), calc_size);
    DEBUG(printf("Core %u: A: %p, B: %p, Size: %u \n", cid, a + (calc_idx >> 3), b + (calc_idx >> 3), calc_size), DBG_LVL_INFO);
    calc_idx += CHUNCK_SIZE;

  } while (load_idx < end_idx);

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // Final reduction
  double acc = 0.0;
  if (cid == 0) {
    for (unsigned int i = 0; i < num_cores; ++i)
      acc += result[i];
    result[0] = acc;
  }

  snrt_cluster_hw_barrier();
