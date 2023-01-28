@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [4096], []),
             B: Buffer(B_2: Pointer(float16), float16, [4096], []),
             C: Buffer(C_2: Pointer(float32), float32, [4096], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [64, 64], []), B_1: B_3: Buffer(B_2, float16, [64, 64], []), C_1: C_3: Buffer(C_2, float32, [64, 64], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [4096]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [4096]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [4096]), storage_scope = wmma.matrix_b {
    for (ax0: int32, 0, 64) {
      for (ax1: int32, 0, 64) {
        let cse_var_1: int32 = ((ax0*64) + ax1)
        A.shared_1: Buffer(A.shared, float16, [4096], [], scope="shared")[cse_var_1] = A[cse_var_1]
      }
    }
    for (ax0_1: int32, 0, 64) {
      for (ax1_1: int32, 0, 64) {
        let cse_var_2: int32 = ((ax0_1*64) + ax1_1)
        A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [4096], [], scope="wmma.matrix_a")[cse_var_2] = A.shared_1[cse_var_2]
      }
    }
    for (ax0_2: int32, 0, 64) {
      for (ax1_2: int32, 0, 64) {
        let cse_var_3: int32 = ((ax0_2*64) + ax1_2)
        A.shared_2: Buffer(A.shared, float16, [4096], [], scope="shared")[cse_var_3] = B[cse_var_3]
      }
    }
    for (ax0_3: int32, 0, 64) {
      for (ax1_3: int32, 0, 64) {
        let cse_var_4: int32 = ((ax0_3*64) + ax1_3)
        B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [4096], [], scope="wmma.matrix_b")[cse_var_4] = A.shared_2[cse_var_4]
      }
    }
    attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 1;
    allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [4096]), storage_scope = wmma.accumulator;
    attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1;
    attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1 {
      for (ii.c.outer.init: int32, 0, 4) {
        for (jj.c.outer.init: int32, 0, 4) {
          for (ii.c.inner.init: int32, 0, 16) {
            for (jj.c.inner.init: int32, 0, 16) {
              C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [4096], [], scope="wmma.accumulator")[((((ii.c.outer.init*1024) + (ii.c.inner.init*64)) + (jj.c.outer.init*16)) + jj.c.inner.init)] = 0f32
            }
          }
        }
      }
      for (rk.outer.outer: int32, 0, 4) {
        for (ii.c.outer: int32, 0, 4) {
          for (jj.c.outer: int32, 0, 4) {
            for (ii.c.inner: int32, 0, 16) {
              for (jj.c.inner: int32, 0, 16) {
                for (rk.inner: int32, 0, 16) {
                  let cse_var_7: int32 = (jj.c.outer*16)
                  let cse_var_6: int32 = ((ii.c.outer*1024) + (ii.c.inner*64))
                  let cse_var_5: int32 = ((cse_var_6 + cse_var_7) + jj.c.inner)
                  C.wmma.accumulator_1[cse_var_5] = (C.wmma.accumulator_1[cse_var_5] + (cast(float32, A.shared.wmma.matrix_a_1[((cse_var_6 + (rk.outer.outer*16)) + rk.inner)])*cast(float32, B.shared.wmma.matrix_b_1[((((rk.outer.outer*1024) + (rk.inner*64)) + cse_var_7) + jj.c.inner)])))
                }
              }
            }
          }
        }
      }
      for (ii.inner.outer.inner: int32, 0, 4) {
        for (jj.inner.outer.inner: int32, 0, 4) {
          for (ii.inner.inner: int32, 0, 16) {
            for (jj.inner.inner: int32, 0, 16) {
              let cse_var_8: int32 = ((((ii.inner.outer.inner*1024) + (ii.inner.inner*64)) + (jj.inner.outer.inner*16)) + jj.inner.inner)
              C[cse_var_8] = C.wmma.accumulator_1[cse_var_8]
            }
          }
        }
      }
    }
  }
}

