@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [256], []),
             B: Buffer(B_2: Pointer(float16), float16, [256], []),
             C: Buffer(C_2: Pointer(float32), float32, [256], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [16, 16], []), B_1: B_3: Buffer(B_2, float16, [16, 16], []), C_1: C_3: Buffer(C_2, float32, [16, 16], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [1808]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float16), float16, [256]), storage_scope = shared {
    for (ax0: int32, 0, 113) {
      if @tir.likely((((((threadIdx.z: int32*4) + (threadIdx.y: int32*2)) + floordiv(threadIdx.x: int32, 16)) + ax0) < 16), dtype=bool) {
        for (ax1: int32, 0, 16) {
          let cse_var_1: int32 = (ax0*16)
          A.shared_1: Buffer(A.shared, float16, [1808], [], scope="shared")[(cse_var_1 + ax1)] = A[(((((threadIdx.z*64) + (threadIdx.y*32)) + (floordiv(threadIdx.x, 16)*16)) + cse_var_1) + ax1)]
        }
      }
    }
    for (ax0_1: int32, 0, 16) {
      for (ax1_1: int32, 0, 16) {
        let cse_var_2: int32 = ((ax0_1*16) + ax1_1)
        B.shared_1: Buffer(B.shared, float16, [256], [], scope="shared")[cse_var_2] = B[cse_var_2]
      }
    }
    attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 1;
    allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [1808]), storage_scope = wmma.accumulator;
    allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [1808]), storage_scope = wmma.matrix_a;
    allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [256]), storage_scope = wmma.matrix_b;
    attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1 {
      for (ii.c.outer.init: int32, 0, 8) {
        for (ii.c.inner.init: int32, 0, 16) {
          if @tir.likely((((ii.c.outer.init*16) + ii.c.inner.init) < 113), dtype=bool) {
            for (jj.c.inner.init: int32, 0, 16) {
              C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [1808], [], scope="wmma.accumulator")[(((ii.c.outer.init*256) + (ii.c.inner.init*16)) + jj.c.inner.init)] = 0f32
            }
          }
        }
      }
      for (ax0.outer: int32, 0, 8) {
        for (ax0.inner: int32, 0, 16) {
          if @tir.likely((((ax0.outer*16) + ax0.inner) < 113), dtype=bool) {
            if @tir.likely(((((((ax0.outer*16) + (threadIdx.z*4)) + (threadIdx.y*2)) + floordiv(threadIdx.x, 16)) + ax0.inner) < 16), dtype=bool) {
              for (ax1.inner: int32, 0, 16) {
                let cse_var_3: int32 = (((ax0.outer*256) + (ax0.inner*16)) + ax1.inner)
                A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [1808], [], scope="wmma.matrix_a")[cse_var_3] = A.shared_1[cse_var_3]
              }
            }
          }
        }
      }
      for (ax0.inner_1: int32, 0, 16) {
        for (ax1.inner_1: int32, 0, 16) {
          let cse_var_4: int32 = ((ax0.inner_1*16) + ax1.inner_1)
          B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [256], [], scope="wmma.matrix_b")[cse_var_4] = B.shared_1[cse_var_4]
        }
      }
      for (ii.c.outer: int32, 0, 8) {
        for (ii.c.inner: int32, 0, 16) {
          if @tir.likely((((ii.c.outer*16) + ii.c.inner) < 113), dtype=bool) {
            if @tir.likely(((((((ii.c.outer*16) + (threadIdx.z*4)) + (threadIdx.y*2)) + floordiv(threadIdx.x, 16)) + ii.c.inner) < 16), dtype=bool) {
              for (jj.c.inner: int32, 0, 16) {
                for (rk.inner: int32, 0, 16) {
                  let cse_var_6: int32 = ((ii.c.outer*256) + (ii.c.inner*16))
                  let cse_var_5: int32 = (cse_var_6 + jj.c.inner)
                  C.wmma.accumulator_1[cse_var_5] = (C.wmma.accumulator_1[cse_var_5] + (cast(float32, A.shared.wmma.matrix_a_1[(cse_var_6 + rk.inner)])*cast(float32, B.shared.wmma.matrix_b_1[((rk.inner*16) + jj.c.inner)])))
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
        if @tir.likely((((((ii.inner.jj.inner.fused.outer.outer.outer.outer*256) + (threadIdx.z*64)) + (threadIdx.y*32)) + threadIdx.x) < 256), dtype=bool) {
          for (ii.inner.jj.inner.fused.inner.s: int32, 0, 8) {
            if @tir.likely((floormod(threadIdx.x, 16) < 2), dtype=bool) {
              let cse_var_7: int32 = (ii.inner.jj.inner.fused.outer.outer.outer.outer*256)
              C[(((((cse_var_7 + (threadIdx.z*64)) + (threadIdx.y*32)) + (floordiv(threadIdx.x, 16)*16)) + (floormod(threadIdx.x, 16)*8)) + ii.inner.jj.inner.fused.inner.s)] = C.wmma.accumulator_1[((cse_var_7 + (floormod(threadIdx.x, 16)*8)) + ii.inner.jj.inner.fused.inner.s)]
            }
          }
        }
      }
    }
  }
}

