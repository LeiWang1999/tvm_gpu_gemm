@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [268435456], []),
             B: Buffer(B_2: Pointer(float32), float32, [268435456], []),
             C: Buffer(C_2: Pointer(float32), float32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [16384, 16384], []), B_1: B_3: Buffer(B_2, float32, [16384, 16384], []), C_1: C_3: Buffer(C_2, float32, [16384, 16384], [])} {
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 128;
  allocate(C.local: Pointer(local float32), float32, [64]), storage_scope = local;
  allocate(A.shared: Pointer(shared float32), float32, [2048]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float32), float32, [2048]), storage_scope = shared;
  allocate(A.shared.local: Pointer(local float32), float32, [8]), storage_scope = local;
  allocate(B.shared.local: Pointer(local float32), float32, [8]), storage_scope = local;
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 128;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 16;
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
    for (ii.c.init: int32, 0, 8) {
      for (jj.c.init: int32, 0, 8) {
        C.local_1: Buffer(C.local, float32, [64], [], scope="local")[((ii.c.init*8) + jj.c.init)] = 0f32
      }
    }
    for (k.outer: int32, 0, 1024) {
      for (ax0: int32, 0, 16) {
        for (ax1: int32, 0, 128) {
          A.shared_1: Buffer(A.shared, float32, [2048], [], scope="shared")[((ax0*128) + ax1)] = A[((((k.outer*262144) + (ax0*16384)) + (blockIdx.y*128)) + ax1)]
        }
      }
      for (ax0_1: int32, 0, 16) {
        for (ax1_1: int32, 0, 128) {
          B.shared_1: Buffer(B.shared, float32, [2048], [], scope="shared")[((ax0_1*128) + ax1_1)] = B[((((k.outer*262144) + (ax0_1*16384)) + (blockIdx.x*128)) + ax1_1)]
        }
      }
      for (k.inner: int32, 0, 16) {
        for (ax1_2: int32, 0, 8) {
          A.shared.local_1: Buffer(A.shared.local, float32, [8], [], scope="local", align=32)[ax1_2] = A.shared_1[(((k.inner*128) + (threadIdx.y*8)) + ax1_2)]
        }
        for (ax1_3: int32, 0, 8) {
          B.shared.local_1: Buffer(B.shared.local, float32, [8], [], scope="local", align=32)[ax1_3] = B.shared_1[(((k.inner*128) + (threadIdx.x*8)) + ax1_3)]
        }
        for (ii.c: int32, 0, 8) {
          for (jj.c: int32, 0, 8) {
            let cse_var_1: int32 = ((ii.c*8) + jj.c)
            C.local_1[cse_var_1] = (C.local_1[cse_var_1] + (A.shared.local_1[jj.c]*B.shared.local_1[ii.c]))
          }
        }
      }
    }
    for (ii.inner.inner: int32, 0, 8) {
      for (jj.inner.inner: int32, 0, 8) {
        C[((((((blockIdx.x*2097152) + (threadIdx.x*131072)) + (ii.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*8)) + jj.inner.inner)] = C.local_1[((ii.inner.inner*8) + jj.inner.inner)]
      }
    }
  }
}

