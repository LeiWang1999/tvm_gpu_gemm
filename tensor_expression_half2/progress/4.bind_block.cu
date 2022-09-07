@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [268435456], []),
             B: Buffer(B_2: Pointer(float16), float16, [268435456], []),
             C: Buffer(C_2: Pointer(float16), float16, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [16384, 16384], []), B_1: B_3: Buffer(B_2, float16, [16384, 16384], []), C_1: C_3: Buffer(C_2, float16, [16384, 16384], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [2097152]), storage_scope = shared;
  allocate(A.shared.local: Pointer(local float16), float16, [2097152]), storage_scope = local;
  allocate(B.shared.local: Pointer(local float16), float16, [2097152]), storage_scope = local;
  allocate(C.local: Pointer(local float16), float16, [16384]), storage_scope = local {
    for (ax0: int32, 0, 16384) {
      for (ax1: int32, 0, 128) {
        A.shared_1: Buffer(A.shared, float16, [2097152], [], scope="shared")[((ax0*128) + ax1)] = A[(((ax0*16384) + (blockIdx.y: int32*128)) + ax1)]
      }
    }
    for (ax0_1: int32, 0, 16384) {
      for (ax1_1: int32, 0, 128) {
        let cse_var_1: int32 = ((ax0_1*128) + ax1_1)
        A.shared.local_1: Buffer(A.shared.local, float16, [2097152], [], scope="local")[cse_var_1] = A.shared_1[cse_var_1]
      }
    }
    for (ax0_2: int32, 0, 16384) {
      for (ax1_2: int32, 0, 128) {
        A.shared_2: Buffer(A.shared, float16, [2097152], [], scope="shared")[((ax0_2*128) + ax1_2)] = B[(((ax0_2*16384) + (blockIdx.x: int32*128)) + ax1_2)]
      }
    }
    for (ax0_3: int32, 0, 16384) {
      for (ax1_3: int32, 0, 128) {
        let cse_var_2: int32 = ((ax0_3*128) + ax1_3)
        B.shared.local_1: Buffer(B.shared.local, float16, [2097152], [], scope="local")[cse_var_2] = A.shared_2[cse_var_2]
      }
    }
    for (ii.c: int32, 0, 128) {
      for (jj.c: int32, 0, 128) {
        C.local_1: Buffer(C.local, float16, [16384], [], scope="local")[((ii.c*128) + jj.c)] = 0f16
        for (k: int32, 0, 16384) {
          let cse_var_4: int32 = (k*128)
          let cse_var_3: int32 = ((ii.c*128) + jj.c)
          C.local_1[cse_var_3] = (C.local_1[cse_var_3] + (A.shared.local_1[(cse_var_4 + jj.c)]*B.shared.local_1[(cse_var_4 + ii.c)]))
        }
      }
    }
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 128;
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 128;
    for (ii.inner: int32, 0, 128) {
      for (jj.inner: int32, 0, 128) {
        C[((((blockIdx.x*2097152) + (ii.inner*16384)) + (blockIdx.y*128)) + jj.inner)] = C.local_1[((ii.inner*128) + jj.inner)]
      }
    }
  }
}

