@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [268435456], []),
             B: Buffer(B_2: Pointer(float32), float32, [268435456], []),
             C: Buffer(C_2: Pointer(float32), float32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [16384, 16384], []), B_1: B_3: Buffer(B_2, float32, [16384, 16384], []), C_1: C_3: Buffer(C_2, float32, [16384, 16384], [])} {
  allocate(A.shared: Pointer(shared float32), float32, [65536]), storage_scope = shared;
  allocate(A.shared.local: Pointer(local float32), float32, [65536]), storage_scope = local;
  allocate(B.shared.local: Pointer(local float32), float32, [65536]), storage_scope = local {
    for (ax0: int32, 0, 16384) {
      for (ax1: int32, 0, 4) {
        A.shared_1: Buffer(A.shared, float32, [65536], [], scope="shared")[((ax0*4) + ax1)] = A[(((((ax0*16384) + (blockIdx.y: int32*128)) + (vy: int32*64)) + (threadIdx.y: int32*4)) + ax1)]
      }
    }
    for (ax0_1: int32, 0, 16384) {
      for (ax1_1: int32, 0, 4) {
        let cse_var_1: int32 = ((ax0_1*4) + ax1_1)
        A.shared.local_1: Buffer(A.shared.local, float32, [65536], [], scope="local")[cse_var_1] = A.shared_1[cse_var_1]
      }
    }
    for (ax0_2: int32, 0, 16384) {
      for (ax1_2: int32, 0, 4) {
        A.shared_2: Buffer(A.shared, float32, [65536], [], scope="shared")[((ax0_2*4) + ax1_2)] = B[(((((ax0_2*16384) + (blockIdx.x: int32*128)) + (vx: int32*64)) + (threadIdx.x: int32*4)) + ax1_2)]
      }
    }
    for (ax0_3: int32, 0, 16384) {
      for (ax1_3: int32, 0, 4) {
        let cse_var_2: int32 = ((ax0_3*4) + ax1_3)
        B.shared.local_1: Buffer(B.shared.local, float32, [65536], [], scope="local")[cse_var_2] = A.shared_2[cse_var_2]
      }
    }
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 128;
    allocate(C.local: Pointer(local float32), float32, [16]), storage_scope = local;
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 128;
    attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 16;
    attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
      for (ii.c: int32, 0, 4) {
        for (jj.c: int32, 0, 4) {
          C.local_1: Buffer(C.local, float32, [16], [], scope="local", align=64)[((ii.c*4) + jj.c)] = 0f32
          for (k: int32, 0, 16384) {
            let cse_var_4: int32 = (k*4)
            let cse_var_3: int32 = ((ii.c*4) + jj.c)
            C.local_1[cse_var_3] = (C.local_1[cse_var_3] + (A.shared.local_1[(cse_var_4 + jj.c)]*B.shared.local_1[(cse_var_4 + ii.c)]))
          }
        }
      }
      for (ii.inner.inner.inner: int32, 0, 4) {
        for (jj.inner.inner.inner: int32, 0, 4) {
          let cse_var_5: int32 = ((ii.inner.inner.inner*4) + jj.inner.inner.inner)
           {
            C[((((((blockIdx.x*2097152) + (threadIdx.x*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*4)) + jj.inner.inner.inner)] = C.local_1[cse_var_5]
            C[(((((((blockIdx.x*2097152) + (threadIdx.x*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*4)) + jj.inner.inner.inner) + 64)] = C.local_1[cse_var_5]
            C[(((((((blockIdx.x*2097152) + (threadIdx.x*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*4)) + jj.inner.inner.inner) + 1048576)] = C.local_1[cse_var_5]
            C[(((((((blockIdx.x*2097152) + (threadIdx.x*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*4)) + jj.inner.inner.inner) + 1048640)] = C.local_1[cse_var_5]
          }
        }
      }
    }
  }
}

