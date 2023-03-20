@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [65536], []),
             B: Buffer(B_2: Pointer(float16), float16, [65536], []),
             C: Buffer(C_2: Pointer(float32), float32, [65536], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [256, 256], []), B_1: B_3: Buffer(B_2, float16, [256, 256], []), C_1: C_3: Buffer(C_2, float32, [256, 256], [])} {
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 2;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [2048]), storage_scope = wmma.accumulator;
  allocate(A.shared: Pointer(shared float16), float16, [2048]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float16), float16, [1024]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [1024]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [512]), storage_scope = wmma.matrix_b;
  allocate(C.wmma.accumulator.shared: Pointer(shared float32), float32, [8192]), storage_scope = shared;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 4 {
    attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2 {
      for (ii.c.outer.init: int32, 0, 2) {
        for (jj.c.outer.init: int32, 0, 4) {
          for (ii.c.inner.init: int32, 0, 32) {
            for (jj.c.inner.init: int32, 0, 8) {
              C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [2048], [], scope="wmma.accumulator")[((((ii.c.outer.init*1024) + (ii.c.inner.init*32)) + (jj.c.outer.init*8)) + jj.c.inner.init)] = 0f32
            }
          }
        }
      }
      for (rk.outer.outer: int32, 0, 16) {
        for (ax0.ax1.fused.outer.outer.outer: int32, 0, 16) {
          attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
          A.shared_1: Buffer(A.shared, float16, [2048], [], scope="shared")[((((ax0.ax1.fused.outer.outer.outer*128) + (threadIdx.z*64)) + (threadIdx.y*32)) + threadIdx.x)] = A[(((((((blockIdx.y*32768) + (ax0.ax1.fused.outer.outer.outer*2048)) + (threadIdx.z*1024)) + (threadIdx.y*512)) + (floordiv(threadIdx.x, 16)*256)) + (rk.outer.outer*16)) + floormod(threadIdx.x, 16))]
        }
        for (ax0.ax1.fused.outer.outer.outer_1: int32, 0, 8) {
          attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
          B.shared_1: Buffer(B.shared, float16, [1024], [], scope="shared")[((((ax0.ax1.fused.outer.outer.outer_1*128) + (threadIdx.z*64)) + (threadIdx.y*32)) + threadIdx.x)] = B[(((((((blockIdx.x*16384) + (ax0.ax1.fused.outer.outer.outer_1*2048)) + (threadIdx.z*1024)) + (threadIdx.y*512)) + (floordiv(threadIdx.x, 16)*256)) + (rk.outer.outer*16)) + floormod(threadIdx.x, 16))]
        }
        for (ax0.outer: int32, 0, 2) {
          for (ax0.inner: int32, 0, 32) {
            for (ax1.inner: int32, 0, 16) {
              let cse_var_2: int32 = (ax0.outer*512)
              let cse_var_1: int32 = (ax0.inner*16)
              A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [1024], [], scope="wmma.matrix_a")[((cse_var_2 + cse_var_1) + ax1.inner)] = A.shared_1[((((threadIdx.z*1024) + cse_var_2) + cse_var_1) + ax1.inner)]
            }
          }
        }
        for (ax0.outer_1: int32, 0, 4) {
          for (ax0.inner_1: int32, 0, 8) {
            for (ax1.inner_1: int32, 0, 16) {
              let cse_var_4: int32 = (ax0.outer_1*128)
              let cse_var_3: int32 = (ax0.inner_1*16)
              B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [512], [], scope="wmma.matrix_b")[((cse_var_4 + cse_var_3) + ax1.inner_1)] = B.shared_1[((((threadIdx.y*512) + cse_var_4) + cse_var_3) + ax1.inner_1)]
            }
          }
        }
        for (ii.c.outer: int32, 0, 2) {
          for (jj.c.outer: int32, 0, 4) {
            for (ii.c.inner: int32, 0, 32) {
              for (jj.c.inner: int32, 0, 8) {
                for (rk.inner: int32, 0, 16) {
                  let cse_var_5: int32 = ((((ii.c.outer*1024) + (ii.c.inner*32)) + (jj.c.outer*8)) + jj.c.inner)
                  C.wmma.accumulator_1[cse_var_5] = (C.wmma.accumulator_1[cse_var_5] + (cast(float32, A.shared.wmma.matrix_a_1[(((ii.c.outer*512) + (ii.c.inner*16)) + rk.inner)])*cast(float32, B.shared.wmma.matrix_b_1[(((jj.c.outer*128) + (jj.c.inner*16)) + rk.inner)])))
                }
              }
            }
          }
        }
      }
      for (ax0.outer.inner: int32, 0, 2) {
        for (ax1.outer.inner: int32, 0, 4) {
          for (ax0.inner_2: int32, 0, 32) {
            for (ax1.inner_2: int32, 0, 8) {
              let cse_var_6: int32 = (ax1.outer.inner*8)
              C.wmma.accumulator.shared_1: Buffer(C.wmma.accumulator.shared, float32, [8192], [], scope="shared")[((((((threadIdx.z*4096) + (ax0.outer.inner*2048)) + (ax0.inner_2*64)) + (threadIdx.y*32)) + cse_var_6) + ax1.inner_2)] = C.wmma.accumulator_1[((((ax0.outer.inner*1024) + (ax0.inner_2*32)) + cse_var_6) + ax1.inner_2)]
            }
          }
        }
      }
    }
    for (ii.inner.jj.inner.fused.outer.outer.outer: int32, 0, 64) {
      attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2;
      attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
      attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
      C[((((((blockIdx.y*32768) + (ii.inner.jj.inner.fused.outer.outer.outer*512)) + (threadIdx.z*256)) + (blockIdx.x*64)) + (threadIdx.y*32)) + threadIdx.x)] = C.wmma.accumulator.shared_1[((((ii.inner.jj.inner.fused.outer.outer.outer*128) + (threadIdx.z*64)) + (threadIdx.y*32)) + threadIdx.x)]
    }
  }
}

