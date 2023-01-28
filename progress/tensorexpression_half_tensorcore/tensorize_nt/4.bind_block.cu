@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [8192], []),
             B: Buffer(B_2: Pointer(float16), float16, [16384], []),
             C: Buffer(C_2: Pointer(float32), float32, [32768], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [128, 64], []), B_1: B_3: Buffer(B_2, float16, [256, 64], []), C_1: C_3: Buffer(C_2, float32, [128, 256], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [1024]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [1024]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [1024]), storage_scope = wmma.matrix_b;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [256]), storage_scope = wmma.accumulator {
    for (ax0: int32, 0, 16) {
      for (ax1: int32, 0, 64) {
        let cse_var_1: int32 = (ax0*64)
        A.shared_1: Buffer(A.shared, float16, [1024], [], scope="shared")[(cse_var_1 + ax1)] = A[(((blockIdx.y: int32*1024) + cse_var_1) + ax1)]
      }
    }
    for (ax0_1: int32, 0, 16) {
      for (ax1_1: int32, 0, 64) {
        let cse_var_2: int32 = ((ax0_1*64) + ax1_1)
        A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [1024], [], scope="wmma.matrix_a")[cse_var_2] = A.shared_1[cse_var_2]
      }
    }
    for (ax0_2: int32, 0, 16) {
      for (ax1_2: int32, 0, 64) {
        let cse_var_3: int32 = (ax0_2*64)
        A.shared_2: Buffer(A.shared, float16, [1024], [], scope="shared")[(cse_var_3 + ax1_2)] = B[(((blockIdx.x: int32*1024) + cse_var_3) + ax1_2)]
      }
    }
    for (ax0_3: int32, 0, 16) {
      for (ax1_3: int32, 0, 64) {
        let cse_var_4: int32 = ((ax0_3*64) + ax1_3)
        B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [1024], [], scope="wmma.matrix_b")[cse_var_4] = A.shared_2[cse_var_4]
      }
    }
    for (ii.c: int32, 0, 16) {
      for (jj.c: int32, 0, 16) {
        C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [256], [], scope="wmma.accumulator")[((ii.c*16) + jj.c)] = 0f32
        for (rk: int32, 0, 64) {
          let cse_var_5: int32 = ((ii.c*16) + jj.c)
          C.wmma.accumulator_1[cse_var_5] = (C.wmma.accumulator_1[cse_var_5] + (cast(float32, A.shared.wmma.matrix_a_1[((ii.c*64) + rk)])*cast(float32, B.shared.wmma.matrix_b_1[((jj.c*64) + rk)])))
        }
      }
    }
    for (ax0_4: int32, 0, 16) {
      for (ax1_4: int32, 0, 16) {
        let cse_var_6: int32 = ((ax0_4*16) + ax1_4)
        A.shared_3: Buffer(A.shared, float32, [256], [], scope="shared")[cse_var_6] = C.wmma.accumulator_1[cse_var_6]
      }
    }
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 8;
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 16;
    attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
    attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    C[ramp(((((blockIdx.y*4096) + (floordiv(threadIdx.x, 2)*256)) + (blockIdx.x*16)) + (floormod(threadIdx.x, 2)*8)), 1, 8)] = A.shared_3[ramp((threadIdx.x*8), 1, 8)]
  }
}

