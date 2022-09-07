@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [268435456], []),
             B: Buffer(B_2: Pointer(float32), float32, [268435456], []),
             C: Buffer(C_2: Pointer(float32), float32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [16384, 16384], []), B_1: B_3: Buffer(B_2, float32, [16384, 16384], []), C_1: C_3: Buffer(C_2, float32, [16384, 16384], [])} {
  allocate(A.shared: Pointer(shared float32), float32, [131072]), storage_scope = shared;
  allocate(A.shared.local: Pointer(local float32), float32, [131072]), storage_scope = local;
  allocate(B.shared.local: Pointer(local float32), float32, [131072]), storage_scope = local;
  allocate(C.local: Pointer(local float32), float32, [64]), storage_scope = local {
    for (ax0: int32, 0, 16384) {
      for (ax1: int32, 0, 8) {
        A.shared_1: Buffer(A.shared, float32, [131072], [], scope="shared")[((ax0*8) + ax1)] = A[((((ax0*16384) + (blockIdx.y: int32*128)) + (threadIdx.y: int32*8)) + ax1)]
      }
    }
    for (ax0_1: int32, 0, 16384) {
      for (ax1_1: int32, 0, 8) {
        let cse_var_1: int32 = ((ax0_1*8) + ax1_1)
        A.shared.local_1: Buffer(A.shared.local, float32, [131072], [], scope="local")[cse_var_1] = A.shared_1[cse_var_1]
      }
    }
    for (ax0_2: int32, 0, 16384) {
      for (ax1_2: int32, 0, 8) {
        A.shared_2: Buffer(A.shared, float32, [131072], [], scope="shared")[((ax0_2*8) + ax1_2)] = B[((((ax0_2*16384) + (blockIdx.x: int32*128)) + (threadIdx.x: int32*8)) + ax1_2)]
      }
    }
    for (ax0_3: int32, 0, 16384) {
      for (ax1_3: int32, 0, 8) {
        let cse_var_2: int32 = ((ax0_3*8) + ax1_3)
        B.shared.local_1: Buffer(B.shared.local, float32, [131072], [], scope="local")[cse_var_2] = A.shared_2[cse_var_2]
      }
    }
    for (ii.c: int32, 0, 8) {
      for (jj.c: int32, 0, 8) {
        C.local_1: Buffer(C.local, float32, [64], [], scope="local")[((ii.c*8) + jj.c)] = 0f32
        for (k: int32, 0, 16384) {
          let cse_var_4: int32 = (k*8)
          let cse_var_3: int32 = ((ii.c*8) + jj.c)
          C.local_1[cse_var_3] = (C.local_1[cse_var_3] + (A.shared.local_1[(cse_var_4 + jj.c)]*B.shared.local_1[(cse_var_4 + ii.c)]))
        }
      }
    }
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 128;
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 128;
    attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 16;
    attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16;
    for (ii.inner.inner: int32, 0, 8) {
      for (jj.inner.inner: int32, 0, 8) {
        C[((((((blockIdx.x*2097152) + (threadIdx.x*131072)) + (ii.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*8)) + jj.inner.inner)] = C.local_1[((ii.inner.inner*8) + jj.inner.inner)]
      }
    }
  }
}

