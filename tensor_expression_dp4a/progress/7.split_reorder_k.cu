@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.shared: Pointer(shared int8), int8, [2097152]), storage_scope = shared;
  allocate(B.shared: Pointer(shared int8), int8, [2097152]), storage_scope = shared {
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
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 128;
    allocate(C.local: Pointer(local int32), int32, [64]), storage_scope = local;
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 128;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 16;
    attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
      for (i.c.init: int32, 0, 8) {
        for (j.c.init: int32, 0, 8) {
          C.local_1: Buffer(C.local, int32, [64], [], scope="local")[((i.c.init*8) + j.c.init)] = 0
        }
      }
      for (k.outer: int32, 0, 512) {
        for (k.inner: int32, 0, 32) {
          for (i.c: int32, 0, 8) {
            for (j.c: int32, 0, 8) {
              let cse_var_4: int32 = (k.outer*32)
              let cse_var_3: int32 = ((i.c*8) + j.c)
              C.local_1[cse_var_3] = (C.local_1[cse_var_3] + (cast(int32, A.shared_1[((((threadIdx.x*131072) + (i.c*16384)) + cse_var_4) + k.inner)])*cast(int32, B.shared_1[((((threadIdx.y*131072) + (j.c*16384)) + cse_var_4) + k.inner)])))
            }
          }
        }
      }
      for (i.inner.inner: int32, 0, 8) {
        for (j.inner.inner: int32, 0, 8) {
          C[((((((blockIdx.x*2097152) + (threadIdx.x*131072)) + (i.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*8)) + j.inner.inner)] = C.local_1[((i.inner.inner*8) + j.inner.inner)]
        }
      }
    }
  }
}

