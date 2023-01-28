@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [4096], []),
             B: Buffer(B_2: Pointer(float16), float16, [4096], []),
             C: Buffer(C_2: Pointer(float32), float32, [4096], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [64, 64], []), B_1: B_3: Buffer(B_2, float16, [64, 64], []), C_1: C_3: Buffer(C_2, float32, [64, 64], [])} {
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 4;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [256]), storage_scope = wmma.accumulator;
  allocate(A.shared: Pointer(shared float16), float16, [256]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float16), float16, [256]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [256]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [256]), storage_scope = wmma.matrix_b;
  allocate(C.wmma.accumulator.shared: Pointer(shared float32), float32, [256]), storage_scope = shared;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 4 {
    attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1 {
      for (ii.c.inner.init: int32, 0, 16) {
        for (jj.c.inner.init: int32, 0, 16) {
          C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [256], [], scope="wmma.accumulator")[((ii.c.inner.init*16) + jj.c.inner.init)] = 0f32
        }
      }
      for (rk.outer.outer: int32, 0, 4) {
        let cse_var_1: int32 = (rk.outer.outer*16)
         {
          attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
          A.shared_1: Buffer(A.shared, float16, [256], [], scope="shared")[ramp((threadIdx.x*8), 1, 8)] = A[ramp(((((blockIdx.y*1024) + (floordiv(threadIdx.x, 2)*64)) + cse_var_1) + (floormod(threadIdx.x, 2)*8)), 1, 8)]
          attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
          B.shared_1: Buffer(B.shared, float16, [256], [], scope="shared")[ramp((threadIdx.x*8), 1, 8)] = B[ramp(((((blockIdx.x*1024) + (floordiv(threadIdx.x, 2)*64)) + cse_var_1) + (floormod(threadIdx.x, 2)*8)), 1, 8)]
          for (ax0.inner: int32, 0, 16) {
            for (ax1.inner: int32, 0, 16) {
              let cse_var_2: int32 = ((ax0.inner*16) + ax1.inner)
              A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [256], [], scope="wmma.matrix_a")[cse_var_2] = A.shared_1[cse_var_2]
            }
          }
          for (ax0.inner_1: int32, 0, 16) {
            for (ax1.inner_1: int32, 0, 16) {
              let cse_var_3: int32 = ((ax0.inner_1*16) + ax1.inner_1)
              B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [256], [], scope="wmma.matrix_b")[cse_var_3] = B.shared_1[cse_var_3]
            }
          }
          for (ii.c.inner: int32, 0, 16) {
            for (jj.c.inner: int32, 0, 16) {
              for (rk.inner: int32, 0, 16) {
                let cse_var_5: int32 = (ii.c.inner*16)
                let cse_var_4: int32 = (cse_var_5 + jj.c.inner)
                C.wmma.accumulator_1[cse_var_4] = (C.wmma.accumulator_1[cse_var_4] + (cast(float32, A.shared.wmma.matrix_a_1[(cse_var_5 + rk.inner)])*cast(float32, B.shared.wmma.matrix_b_1[((jj.c.inner*16) + rk.inner)])))
              }
            }
          }
        }
      }
      for (ax0.inner_2: int32, 0, 16) {
        for (ax1.inner_2: int32, 0, 16) {
          let cse_var_6: int32 = ((ax0.inner_2*16) + ax1.inner_2)
          C.wmma.accumulator.shared_1: Buffer(C.wmma.accumulator.shared, float32, [256], [], scope="shared")[cse_var_6] = C.wmma.accumulator_1[cse_var_6]
        }
      }
    }
    attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
    attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
    attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    C[ramp(((((blockIdx.y*1024) + (floordiv(threadIdx.x, 2)*64)) + (blockIdx.x*16)) + (floormod(threadIdx.x, 2)*8)), 1, 8)] = C.wmma.accumulator.shared_1[ramp((threadIdx.x*8), 1, 8)]
  }
}

