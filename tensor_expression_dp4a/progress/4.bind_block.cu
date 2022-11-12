@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.shared: Pointer(shared int8), int8, [2097152]), storage_scope = shared;
  allocate(B.shared: Pointer(shared int8), int8, [2097152]), storage_scope = shared;
  allocate(C.local: Pointer(local int32), int32, [16384]), storage_scope = local {
    for (ax0: int32, 0, 128) {
      for (ax1: int32, 0, 16384) {
        let cse_var_1: int32 = (ax0*16384)
        A.shared_1: Buffer(A.shared, int8, [2097152], [], scope="shared")[(cse_var_1 + ax1)] = A[(((blockIdx.x: int32*2097152) + cse_var_1) + ax1)]
      }
    }
    for (ax0_1: int32, 0, 128) {
      for (ax1_1: int32, 0, 16384) {
        let cse_var_2: int32 = (ax0_1*16384)
        B.shared_1: Buffer(B.shared, int8, [2097152], [], scope="shared")[(cse_var_2 + ax1_1)] = B[(((blockIdx.y: int32*2097152) + cse_var_2) + ax1_1)]
      }
    }
    for (i.c: int32, 0, 128) {
      for (j.c: int32, 0, 128) {
        C.local_1: Buffer(C.local, int32, [16384], [], scope="local")[((i.c*128) + j.c)] = 0
        for (k: int32, 0, 16384) {
          let cse_var_3: int32 = ((i.c*128) + j.c)
          C.local_1[cse_var_3] = (C.local_1[cse_var_3] + (cast(int32, A.shared_1[((i.c*16384) + k)])*cast(int32, B.shared_1[((j.c*16384) + k)])))
        }
      }
    }
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 128;
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 128;
    for (i.inner: int32, 0, 128) {
      for (j.inner: int32, 0, 128) {
        C[((((blockIdx.x*2097152) + (i.inner*16384)) + (blockIdx.y*128)) + j.inner)] = C.local_1[((i.inner*128) + j.inner)]
      }
    }
  }
}

