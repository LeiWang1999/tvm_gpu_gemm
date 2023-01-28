@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [256], []),
             B: Buffer(B_2: Pointer(float16), float16, [256], []),
             C: Buffer(C_2: Pointer(float32), float32, [256], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [16, 16], []), B_1: B_3: Buffer(B_2, float16, [16, 16], []), C_1: C_3: Buffer(C_2, float32, [16, 16], [])} {
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 1;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [256]), storage_scope = wmma.accumulator;
  allocate(A.shared: Pointer(shared float16), float16, [256]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float16), float16, [256]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [256]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [256]), storage_scope = wmma.matrix_b;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1;
  attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 1;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1 {
    @tir.tvm_fill_fragment(C.wmma.accumulator, 16, 16, 16, 0, 0f32, dtype=handle)
    attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    A.shared_1: Buffer(A.shared, float16, [256], [], scope="shared")[ramp((threadIdx.x*8), 1, 8)] = A[ramp((threadIdx.x*8), 1, 8)]
    attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    B.shared_1: Buffer(B.shared, float16, [256], [], scope="shared")[ramp((threadIdx.x*8), 1, 8)] = B[ramp((threadIdx.x*8), 1, 8)]
    @tir.tvm_load_matrix_sync(A.shared.wmma.matrix_a, 16, 16, 16, 0, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), A.shared, 0, 256, 1, dtype=handle), 16, "row_major", dtype=handle)
    @tir.tvm_load_matrix_sync(B.shared.wmma.matrix_b, 16, 16, 16, 0, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), B.shared, 0, 256, 1, dtype=handle), 16, "row_major", dtype=handle)
    @tir.tvm_mma_sync(C.wmma.accumulator, 0, A.shared.wmma.matrix_a, 0, B.shared.wmma.matrix_b, 0, C.wmma.accumulator, 0, dtype=handle)
    @tir.tvm_store_matrix_sync(C.wmma.accumulator, 16, 16, 16, 0, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), C_2, 0, 256, 2, dtype=handle), 16, "row_major", dtype=handle)
  }
}

