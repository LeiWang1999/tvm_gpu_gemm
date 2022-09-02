@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [268435456], []),
             B: Buffer(B_2: Pointer(float32), float32, [268435456], []),
             C: Buffer(C_2: Pointer(float32), float32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [16384, 16384], []), B_1: B_3: Buffer(B_2, float32, [16384, 16384], []), C_1: C_3: Buffer(C_2, float32, [16384, 16384], [])} {
  allocate(C.local: Pointer(local float32), float32, [1]), storage_scope = local {
    C.local_1: Buffer(C.local, float32, [1], [], scope="local", align=4)[0] = 0f32
    for (k: int32, 0, 16384) {
      let cse_var_1: int32 = (k*16384)
      C.local_1[0] = (C.local_1[0] + (A[((cse_var_1 + (blockIdx.y: int32*32)) + threadIdx.y: int32)]*B[((cse_var_1 + (blockIdx.x: int32*32)) + threadIdx.x: int32)]))
    }
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 512;
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 512;
    attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 32;
    C[((((blockIdx.x*524288) + (threadIdx.x*16384)) + (blockIdx.y*32)) + threadIdx.y)] = C.local_1[0]
  }
}

