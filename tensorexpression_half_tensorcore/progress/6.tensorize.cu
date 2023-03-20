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
          @tir.tvm_fill_fragment(C.wmma.accumulator, 32, 8, 16, ((ii.c.outer.init*4) + jj.c.outer.init), 0f32, dtype=handle)
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
          @tir.tvm_load_matrix_sync(A.shared.wmma.matrix_a, 32, 8, 16, ax0.outer, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), A.shared, ((threadIdx.z*1024) + (ax0.outer*512)), 512, 1, dtype=handle), 16, "row_major", dtype=handle)
        }
        for (ax0.outer_1: int32, 0, 4) {
          @tir.tvm_load_matrix_sync(B.shared.wmma.matrix_b, 32, 8, 16, ax0.outer_1, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), B.shared, ((threadIdx.y*512) + (ax0.outer_1*128)), 128, 1, dtype=handle), 16, "col_major", dtype=handle)
        }
        for (ii.c.outer: int32, 0, 2) {
          for (jj.c.outer: int32, 0, 4) {
            let cse_var_1: int32 = ((ii.c.outer*4) + jj.c.outer)
            @tir.tvm_mma_sync(C.wmma.accumulator, cse_var_1, A.shared.wmma.matrix_a, ii.c.outer, B.shared.wmma.matrix_b, jj.c.outer, C.wmma.accumulator, cse_var_1, dtype=handle)
          }
        }
      }
      for (ax0.outer.inner: int32, 0, 2) {
        for (ax1.outer.inner: int32, 0, 4) {
          @tir.tvm_store_matrix_sync(C.wmma.accumulator, 32, 8, 16, ((ax0.outer.inner*4) + ax1.outer.inner), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), C.wmma.accumulator.shared, ((((threadIdx.z*4096) + (ax0.outer.inner*2048)) + (threadIdx.y*32)) + (ax1.outer.inner*8)), 2048, 2, dtype=handle), 64, "row_major", dtype=handle)
        }
      }
    }
    for (ii.inner.jj.inner.fused.outer.outer.outer: int32, 0, 64) {
      attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2;
      attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
      attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
      C[((((((blockIdx.y*32768) + (ii.inner.jj.inner.fused.outer.outer.outer*512)) + (threadIdx.z*256)) + (blockIdx.x*64)) + (threadIdx.y*32)) + threadIdx.x)] = C.wmma.accumulator.shared_1: Buffer(C.wmma.accumulator.shared, float32, [8192], [], scope="shared")[((((ii.inner.jj.inner.fused.outer.outer.outer*128) + (threadIdx.z*64)) + (threadIdx.y*32)) + threadIdx.x)]
    }
  }
}

