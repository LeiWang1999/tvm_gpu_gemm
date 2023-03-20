@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [65536], []),
             B: Buffer(B_2: Pointer(float16), float16, [65536], []),
             C: Buffer(C_2: Pointer(float32), float32, [65536], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [256, 256], []), B_1: B_3: Buffer(B_2, float16, [256, 256], []), C_1: C_3: Buffer(C_2, float32, [256, 256], [])} {
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 2;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [759]), storage_scope = wmma.accumulator;
  allocate(A.shared: Pointer(shared float16), float16, [48]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float16), float16, [4048]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [48]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [4048]), storage_scope = wmma.matrix_b;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1 {
    for (jj.c.outer.init: int32, 0, 16) {
      for (ii.c.inner.init: int32, 0, 3) {
        for (jj.c.inner.init: int32, 0, 16) {
          if @tir.likely((((jj.c.outer.init*16) + jj.c.inner.init) < 253), dtype=bool) {
            C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [759], [], scope="wmma.accumulator")[(((ii.c.inner.init*253) + (jj.c.outer.init*16)) + jj.c.inner.init)] = 0f32
          }
        }
      }
    }
    for (rk.outer.outer: int32, 0, 16) {
      attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
      attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 4;
      attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2;
      if @tir.likely((threadIdx.x < 6), dtype=bool) {
        A.shared_1: Buffer(A.shared, float16, [48], [], scope="shared", align=64)[(((threadIdx.x*8) + (threadIdx.y*2)) + threadIdx.z)] = A[((((((blockIdx.y*32768) + (threadIdx.x*1024)) + (floordiv(threadIdx.x, 2)*256)) + (threadIdx.z*256)) + (rk.outer.outer*16)) + floormod((((threadIdx.x*8) + (threadIdx.y*2)) + threadIdx.z), 16))]
      }
      for (ax0.ax1.fused.outer.outer.outer: int32, 0, 16) {
        attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
        attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 4;
        attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2;
        if @tir.likely((((((ax0.ax1.fused.outer.outer.outer*256) + (threadIdx.x*8)) + (threadIdx.y*2)) + threadIdx.z) < 4048), dtype=bool) {
          if @tir.likely((((ax0.ax1.fused.outer.outer.outer*16) + floordiv(threadIdx.x, 2)) < 253), dtype=bool) {
            B.shared_1: Buffer(B.shared, float16, [4048], [], scope="shared")[((((ax0.ax1.fused.outer.outer.outer*256) + (threadIdx.x*8)) + (threadIdx.y*2)) + threadIdx.z)] = B[((((rk.outer.outer*4096) + (floordiv(((((ax0.ax1.fused.outer.outer.outer*256) + (threadIdx.x*8)) + (threadIdx.y*2)) + threadIdx.z), 253)*256)) + threadIdx.y) + floormod(((((ax0.ax1.fused.outer.outer.outer*256) + (threadIdx.x*8)) + (threadIdx.y*2)) + threadIdx.z), 253))]
          }
        }
      }
      for (ax0.inner: int32, 0, 3) {
        for (ax1.inner: int32, 0, 16) {
          let cse_var_1: int32 = ((ax0.inner*16) + ax1.inner)
          A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [48], [], scope="wmma.matrix_a", align=64)[cse_var_1] = A.shared_1[cse_var_1]
        }
      }
      for (ax1.outer: int32, 0, 16) {
        for (ax0.inner_1: int32, 0, 16) {
          for (ax1.inner_1: int32, 0, 16) {
            if @tir.likely((((ax1.outer*16) + ax1.inner_1) < 253), dtype=bool) {
              let cse_var_2: int32 = (((ax0.inner_1*253) + (ax1.outer*16)) + ax1.inner_1)
              B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [4048], [], scope="wmma.matrix_b")[cse_var_2] = B.shared_1[cse_var_2]
            }
          }
        }
      }
      for (jj.c.outer: int32, 0, 16) {
        for (ii.c.inner: int32, 0, 3) {
          for (jj.c.inner: int32, 0, 16) {
            if @tir.likely((((jj.c.outer*16) + jj.c.inner) < 253), dtype=bool) {
              for (rk.inner: int32, 0, 16) {
                let cse_var_4: int32 = (jj.c.outer*16)
                let cse_var_3: int32 = (((ii.c.inner*253) + cse_var_4) + jj.c.inner)
                C.wmma.accumulator_1[cse_var_3] = (C.wmma.accumulator_1[cse_var_3] + (cast(float32, A.shared.wmma.matrix_a_1[((ii.c.inner*16) + rk.inner)])*cast(float32, B.shared.wmma.matrix_b_1[(((rk.inner*253) + cse_var_4) + jj.c.inner)])))
              }
            }
          }
        }
      }
    }
    attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2;
    attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 4;
    attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    for (ii.inner.outer.jj.inner.outer.fused.inner: int32, 0, 128) {
      let cse_var_6: int32 = floordiv(ii.inner.outer.jj.inner.outer.fused.inner, 64)
      let cse_var_5: int32 = (floormod(ii.inner.outer.jj.inner.outer.fused.inner, 64)*4)
      C[((((((blockIdx.y*32768) + (threadIdx.x*1024)) + (cse_var_6*512)) + (threadIdx.z*256)) + cse_var_5) + threadIdx.y)] = C.wmma.accumulator_1[((cse_var_6*506) + cse_var_5)]
    }
  }
}

