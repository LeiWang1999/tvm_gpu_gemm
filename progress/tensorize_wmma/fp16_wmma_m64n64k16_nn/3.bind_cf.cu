@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [1048576], []),
             B: Buffer(B_2: Pointer(float16), float16, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [1024, 1024], []), B_1: B_3: Buffer(B_2, float16, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [131072]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [131072]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [16384]), storage_scope = wmma.matrix_b {
    for (ax0: int32, 0, 128) {
      for (ax1: int32, 0, 1024) {
        let cse_var_1: int32 = (ax0*1024)
        A.shared_1: Buffer(A.shared, float16, [131072], [], scope="shared")[(cse_var_1 + ax1)] = A[((((blockIdx.x: int32*262144) + (threadIdx.y: int32*131072)) + cse_var_1) + ax1)]
      }
    }
    for (ax0_1: int32, 0, 128) {
      for (ax1_1: int32, 0, 1024) {
        let cse_var_2: int32 = ((ax0_1*1024) + ax1_1)
        A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [131072], [], scope="wmma.matrix_a")[cse_var_2] = A.shared_1[cse_var_2]
      }
    }
    for (ax0_2: int32, 0, 1024) {
      for (ax1_2: int32, 0, 16) {
        A.shared_2: Buffer(A.shared, float16, [16384], [], scope="shared")[((ax0_2*16) + ax1_2)] = B[((((ax0_2*1024) + (blockIdx.y: int32*64)) + (threadIdx.z: int32*16)) + ax1_2)]
      }
    }
    for (ax0_3: int32, 0, 1024) {
      for (ax1_3: int32, 0, 16) {
        let cse_var_3: int32 = ((ax0_3*16) + ax1_3)
        B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [16384], [], scope="wmma.matrix_b")[cse_var_3] = A.shared_2[cse_var_3]
      }
    }
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 4;
    allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [2048]), storage_scope = wmma.accumulator;
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 16;
    attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
    attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 4 {
      for (ii.c.outer.init: int32, 0, 4) {
        for (jj.c.outer.init: int32, 0, 2) {
          for (ii.c.inner.init: int32, 0, 32) {
            for (jj.c.inner.init: int32, 0, 8) {
              C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [2048], [], scope="wmma.accumulator")[((((ii.c.outer.init*512) + (ii.c.inner.init*16)) + (jj.c.outer.init*8)) + jj.c.inner.init)] = 0f32
            }
          }
        }
      }
      for (rk.outer.outer: int32, 0, 16) {
        for (rk.outer.inner: int32, 0, 4) {
          for (ii.c.outer: int32, 0, 4) {
            for (jj.c.outer: int32, 0, 2) {
              for (ii.c.inner: int32, 0, 32) {
                for (jj.c.inner: int32, 0, 8) {
                  for (rk.inner: int32, 0, 16) {
                    let cse_var_5: int32 = (jj.c.outer*8)
                    let cse_var_4: int32 = ((((ii.c.outer*512) + (ii.c.inner*16)) + cse_var_5) + jj.c.inner)
                    C.wmma.accumulator_1[cse_var_4] = (C.wmma.accumulator_1[cse_var_4] + (cast(float32, A.shared.wmma.matrix_a_1[(((((ii.c.outer*32768) + (ii.c.inner*1024)) + (rk.outer.outer*64)) + (rk.outer.inner*16)) + rk.inner)])*cast(float32, B.shared.wmma.matrix_b_1[(((((rk.outer.outer*1024) + (rk.outer.inner*256)) + (rk.inner*16)) + cse_var_5) + jj.c.inner)])))
                  }
                }
              }
            }
          }
        }
      }
      for (ii.outer.inner: int32, 0, 4) {
        for (jj.outer.inner: int32, 0, 2) {
          for (ii.inner: int32, 0, 32) {
            for (jj.inner: int32, 0, 8) {
              let cse_var_6: int32 = (jj.outer.inner*8)
              C[((((((((blockIdx.x*262144) + (threadIdx.y*131072)) + (ii.outer.inner*32768)) + (ii.inner*1024)) + (blockIdx.y*64)) + (threadIdx.z*16)) + cse_var_6) + jj.inner)] = C.wmma.accumulator_1[((((ii.outer.inner*512) + (ii.inner*16)) + cse_var_6) + jj.inner)]
            }
          }
        }
      }
    }
  }
}

