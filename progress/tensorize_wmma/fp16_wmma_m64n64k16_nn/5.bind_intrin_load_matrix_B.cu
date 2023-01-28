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
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 4;
  attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1 {
    for (ii.c.inner.init: int32, 0, 16) {
      for (jj.c.inner.init: int32, 0, 16) {
        C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [256], [], scope="wmma.accumulator")[((ii.c.inner.init*16) + jj.c.inner.init)] = 0f32
      }
    }
    for (rk.outer.outer: int32, 0, 4) {
      attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
      A.shared_1: Buffer(A.shared, float16, [256], [], scope="shared")[ramp((threadIdx.x*8), 1, 8)] = A[ramp(((((blockIdx.y*1024) + (floordiv(threadIdx.x, 2)*64)) + (rk.outer.outer*16)) + (floormod(threadIdx.x, 2)*8)), 1, 8)]
      attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
      B.shared_1: Buffer(B.shared, float16, [256], [], scope="shared")[ramp((threadIdx.x*8), 1, 8)] = B[ramp(((((rk.outer.outer*1024) + (floordiv(threadIdx.x, 2)*64)) + (blockIdx.x*16)) + (floormod(threadIdx.x, 2)*8)), 1, 8)]
      @tir.tvm_load_matrix_sync(A.shared.wmma.matrix_a, 16, 16, 16, 0, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), A.shared, 0, 256, 1, dtype=handle), 16, "row_major", dtype=handle)
      @tir.tvm_load_matrix_sync(B.shared.wmma.matrix_b, 16, 16, 16, 0, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), B.shared, 0, 256, 1, dtype=handle), 16, "row_major", dtype=handle)
      for (ii.c.inner: int32, 0, 16) {
        for (jj.c.inner: int32, 0, 16) {
          for (rk.inner: int32, 0, 16) {
            let cse_var_2: int32 = (ii.c.inner*16)
            let cse_var_1: int32 = (cse_var_2 + jj.c.inner)
            C.wmma.accumulator_1[cse_var_1] = (C.wmma.accumulator_1[cse_var_1] + (cast(float32, A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [256], [], scope="wmma.matrix_a")[(cse_var_2 + rk.inner)])*cast(float32, B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [256], [], scope="wmma.matrix_b")[((rk.inner*16) + jj.c.inner)])))
          }
        }
      }
    }
    for (ii.inner.inner: int32, 0, 16) {
      for (jj.inner.inner: int32, 0, 16) {
        C[((((blockIdx.y*1024) + (ii.inner.inner*64)) + (blockIdx.x*16)) + jj.inner.inner)] = C.wmma.accumulator_1[((ii.inner.inner*16) + jj.inner.inner)]
      }
    }
  }
}

