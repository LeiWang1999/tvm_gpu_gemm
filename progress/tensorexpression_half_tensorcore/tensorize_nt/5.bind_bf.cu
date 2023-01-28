@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [8192], []),
             B: Buffer(B_2: Pointer(float16), float16, [16384], []),
             C: Buffer(C_2: Pointer(float32), float32, [32768], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [128, 64], []), B_1: B_3: Buffer(B_2, float16, [256, 64], []), C_1: C_3: Buffer(C_2, float32, [128, 256], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [1024]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float16), float16, [1024]), storage_scope = shared {
    for (ax0: int32, 0, 16) {
      for (ax1: int32, 0, 64) {
        let cse_var_1: int32 = (ax0*64)
        A.shared_1: Buffer(A.shared, float16, [1024], [], scope="shared")[(cse_var_1 + ax1)] = A[(((blockIdx.y: int32*1024) + cse_var_1) + ax1)]
      }
    }
    for (ax0_1: int32, 0, 16) {
      for (ax1_1: int32, 0, 64) {
        let cse_var_2: int32 = (ax0_1*64)
        B.shared_1: Buffer(B.shared, float16, [1024], [], scope="shared")[(cse_var_2 + ax1_1)] = B[(((blockIdx.x: int32*1024) + cse_var_2) + ax1_1)]
      }
    }
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 8;
    allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [256]), storage_scope = wmma.accumulator;
    allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [256]), storage_scope = wmma.matrix_a;
    allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [256]), storage_scope = wmma.matrix_b;
    allocate(C.wmma.accumulator.shared: Pointer(shared float32), float32, [256]), storage_scope = shared;
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 16 {
      attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
      attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1 {
        for (ii.c.inner.init: int32, 0, 16) {
          for (jj.c.inner.init: int32, 0, 16) {
            C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [256], [], scope="wmma.accumulator")[((ii.c.inner.init*16) + jj.c.inner.init)] = 0f32
          }
        }
        for (rk.outer.outer: int32, 0, 4) {
          for (ax0.inner: int32, 0, 16) {
            for (ax1.inner: int32, 0, 16) {
              A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [256], [], scope="wmma.matrix_a")[((ax0.inner*16) + ax1.inner)] = A.shared_1[(((ax0.inner*64) + (rk.outer.outer*16)) + ax1.inner)]
            }
          }
          for (ax0.inner_1: int32, 0, 16) {
            for (ax1.inner_1: int32, 0, 16) {
              B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [256], [], scope="wmma.matrix_b")[((ax0.inner_1*16) + ax1.inner_1)] = B.shared_1[(((ax0.inner_1*64) + (rk.outer.outer*16)) + ax1.inner_1)]
            }
          }
          for (ii.c.inner: int32, 0, 16) {
            for (jj.c.inner: int32, 0, 16) {
              for (rk.inner: int32, 0, 16) {
                let cse_var_4: int32 = (ii.c.inner*16)
                let cse_var_3: int32 = (cse_var_4 + jj.c.inner)
                C.wmma.accumulator_1[cse_var_3] = (C.wmma.accumulator_1[cse_var_3] + (cast(float32, A.shared.wmma.matrix_a_1[(cse_var_4 + rk.inner)])*cast(float32, B.shared.wmma.matrix_b_1[((jj.c.inner*16) + rk.inner)])))
              }
            }
          }
        }
        for (ax0.inner_2: int32, 0, 16) {
          for (ax1.inner_2: int32, 0, 16) {
            let cse_var_5: int32 = ((ax0.inner_2*16) + ax1.inner_2)
            C.wmma.accumulator.shared_1: Buffer(C.wmma.accumulator.shared, float32, [256], [], scope="shared")[cse_var_5] = C.wmma.accumulator_1[cse_var_5]
          }
        }
      }
      attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
      attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
      attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
      C[ramp(((((blockIdx.y*4096) + (floordiv(threadIdx.x, 2)*256)) + (blockIdx.x*16)) + (floormod(threadIdx.x, 2)*8)), 1, 8)] = C.wmma.accumulator.shared_1[ramp((threadIdx.x*8), 1, 8)]
    }
  }
}

