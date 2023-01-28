@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [4096], []),
             B: Buffer(B_2: Pointer(float16), float16, [8192], []),
             C: Buffer(C_2: Pointer(float32), float32, [32768], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [128, 32], []), B_1: B_3: Buffer(B_2, float16, [256, 32], []), C_1: C_3: Buffer(C_2, float32, [128, 256], [])} {
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 2;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [4096]), storage_scope = wmma.accumulator;
  allocate(A.shared: Pointer(shared float16), float16, [1024]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float16), float16, [1024]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [1024]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [1024]), storage_scope = wmma.matrix_b;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 4;
  attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1 {
    for (ii.c.outer.init: int32, 0, 4) {
      for (jj.c.outer.init: int32, 0, 4) {
        @tir.tvm_fill_fragment(C.wmma.accumulator, 16, 16, 16, ((ii.c.outer.init*4) + jj.c.outer.init), 0f32, dtype=handle)
      }
    }
    for (rk.outer.outer: int32, 0, 2) {
      for (ax0.ax1.fused.outer.outer.outer.outer: int32, 0, 4) {
        attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
        A.shared_1: Buffer(A.shared, float16, [1024], [], scope="shared")[ramp(((ax0.ax1.fused.outer.outer.outer.outer*256) + (threadIdx.x*8)), 1, 8)] = A[ramp((((((blockIdx.y*2048) + (ax0.ax1.fused.outer.outer.outer.outer*512)) + (floordiv(threadIdx.x, 2)*32)) + (rk.outer.outer*16)) + (floormod(threadIdx.x, 2)*8)), 1, 8)]
      }
      for (ax0.ax1.fused.outer.outer.outer.outer_1: int32, 0, 4) {
        attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
        B.shared_1: Buffer(B.shared, float16, [1024], [], scope="shared")[ramp(((ax0.ax1.fused.outer.outer.outer.outer_1*256) + (threadIdx.x*8)), 1, 8)] = B[ramp((((((blockIdx.x*2048) + (ax0.ax1.fused.outer.outer.outer.outer_1*512)) + (floordiv(threadIdx.x, 2)*32)) + (rk.outer.outer*16)) + (floormod(threadIdx.x, 2)*8)), 1, 8)]
      }
      for (ax0.outer: int32, 0, 4) {
        @tir.tvm_load_matrix_sync(A.shared.wmma.matrix_a, 16, 16, 16, ax0.outer, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), A.shared, (ax0.outer*256), 256, 1, dtype=handle), 16, "row_major", dtype=handle)
      }
      for (ax0.outer_1: int32, 0, 4) {
        @tir.tvm_load_matrix_sync(B.shared.wmma.matrix_b, 16, 16, 16, ax0.outer_1, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), B.shared, (ax0.outer_1*256), 256, 1, dtype=handle), 16, "col_major", dtype=handle)
      }
      for (ii.c.outer: int32, 0, 4) {
        for (jj.c.outer: int32, 0, 4) {
          let cse_var_1: int32 = ((ii.c.outer*4) + jj.c.outer)
          @tir.tvm_mma_sync(C.wmma.accumulator, cse_var_1, A.shared.wmma.matrix_a, ii.c.outer, B.shared.wmma.matrix_b, jj.c.outer, C.wmma.accumulator, cse_var_1, dtype=handle)
        }
      }
    }
    for (ii.inner.outer.inner: int32, 0, 4) {
      for (jj.inner.outer.inner: int32, 0, 4) {
        for (ii.inner.inner: int32, 0, 16) {
          for (jj.inner.inner: int32, 0, 16) {
            let cse_var_2: int32 = (jj.inner.outer.inner*16)
            C[((((((blockIdx.y*16384) + (ii.inner.outer.inner*4096)) + (ii.inner.inner*256)) + (blockIdx.x*64)) + cse_var_2) + jj.inner.inner)] = C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [4096], [], scope="wmma.accumulator")[((((ii.inner.outer.inner*1024) + (ii.inner.inner*64)) + cse_var_2) + jj.inner.inner)]
          }
        }
      }
    }
  }
}

