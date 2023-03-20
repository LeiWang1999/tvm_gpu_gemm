@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [65536], []),
             B: Buffer(B_2: Pointer(float16), float16, [65536], []),
             C: Buffer(C_2: Pointer(float32), float32, [65536], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [256, 256], []), B_1: B_3: Buffer(B_2, float16, [256, 256], []), C_1: C_3: Buffer(C_2, float32, [256, 256], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [28928]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float16), float16, [32768]), storage_scope = shared;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [32768]), storage_scope = wmma.matrix_b {
    for (ax0: int32, 0, 113) {
      for (ax1: int32, 0, 256) {
        let cse_var_1: int32 = (ax0*256)
        A.shared_1: Buffer(A.shared, float16, [28928], [], scope="shared")[(cse_var_1 + ax1)] = A[((((((blockIdx.y: int32*32768) + (threadIdx.z: int32*1024)) + (threadIdx.y: int32*512)) + (floordiv(threadIdx.x: int32, 16)*256)) + cse_var_1) + ax1)]
      }
    }
    for (ax0_1: int32, 0, 256) {
      for (ax1_1: int32, 0, 128) {
        B.shared_1: Buffer(B.shared, float16, [32768], [], scope="shared")[((ax0_1*128) + ax1_1)] = B[(((ax0_1*256) + (blockIdx.x: int32*128)) + ax1_1)]
      }
    }
    for (ax0_2: int32, 0, 256) {
      for (ax1_2: int32, 0, 128) {
        let cse_var_2: int32 = ((ax0_2*128) + ax1_2)
        B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [32768], [], scope="wmma.matrix_b")[cse_var_2] = B.shared_1[cse_var_2]
      }
    }
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 2;
    allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [14464]), storage_scope = wmma.accumulator;
    allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [1808]), storage_scope = wmma.matrix_a;
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 2 {
      for (ii.c.outer.init: int32, 0, 8) {
        for (jj.c.outer.init: int32, 0, 8) {
          for (ii.c.inner.init: int32, 0, 16) {
            if @tir.likely((((ii.c.outer.init*16) + ii.c.inner.init) < 113), dtype=bool) {
              for (jj.c.inner.init: int32, 0, 16) {
                C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [14464], [], scope="wmma.accumulator")[((((ii.c.outer.init*2048) + (ii.c.inner.init*128)) + (jj.c.outer.init*16)) + jj.c.inner.init)] = 0f32
              }
            }
          }
        }
      }
      for (rk.outer.outer: int32, 0, 16) {
        for (ax0.outer: int32, 0, 8) {
          for (ax0.inner: int32, 0, 16) {
            if @tir.likely((((ax0.outer*16) + ax0.inner) < 113), dtype=bool) {
              for (ax1.inner: int32, 0, 16) {
                if @tir.likely((((((((blockIdx.y*128) + (ax0.outer*16)) + (threadIdx.z*4)) + (threadIdx.y*2)) + floordiv(threadIdx.x, 16)) + ax0.inner) < 256), dtype=bool) {
                  A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [1808], [], scope="wmma.matrix_a")[(((ax0.outer*256) + (ax0.inner*16)) + ax1.inner)] = A.shared_1[((((ax0.outer*4096) + (ax0.inner*256)) + (rk.outer.outer*16)) + ax1.inner)]
                }
              }
            }
          }
        }
        for (ii.c.outer: int32, 0, 8) {
          for (jj.c.outer: int32, 0, 8) {
            for (ii.c.inner: int32, 0, 16) {
              if @tir.likely((((ii.c.outer*16) + ii.c.inner) < 113), dtype=bool) {
                for (jj.c.inner: int32, 0, 16) {
                  for (rk.inner: int32, 0, 16) {
                    if @tir.likely((((((((blockIdx.y*128) + (ii.c.outer*16)) + (threadIdx.z*4)) + (threadIdx.y*2)) + floordiv(threadIdx.x, 16)) + ii.c.inner) < 256), dtype=bool) {
                      let cse_var_4: int32 = (jj.c.outer*16)
                      let cse_var_3: int32 = ((((ii.c.outer*2048) + (ii.c.inner*128)) + cse_var_4) + jj.c.inner)
                      C.wmma.accumulator_1[cse_var_3] = (C.wmma.accumulator_1[cse_var_3] + (cast(float32, A.shared.wmma.matrix_a_1[(((ii.c.outer*256) + (ii.c.inner*16)) + rk.inner)])*cast(float32, B.shared.wmma.matrix_b_1[((((rk.outer.outer*2048) + (rk.inner*128)) + cse_var_4) + jj.c.inner)])))
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (ii.inner.jj.inner.fused.outer.outer.outer.outer: int32, 0, 8) {
        attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 4;
        attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
        attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
        C[ramp((((((((blockIdx.y*32768) + (ii.inner.jj.inner.fused.outer.outer.outer.outer*4096)) + (threadIdx.z*1024)) + (threadIdx.y*512)) + (floordiv(threadIdx.x, 16)*256)) + (blockIdx.x*128)) + (floormod(threadIdx.x, 16)*8)), 1, 8)] = C.wmma.accumulator_1[ramp((((ii.inner.jj.inner.fused.outer.outer.outer.outer*2048) + (threadIdx.x*8)) - (floordiv(threadIdx.x, 16)*128)), 1, 8)]
      }
    }
  }
}

