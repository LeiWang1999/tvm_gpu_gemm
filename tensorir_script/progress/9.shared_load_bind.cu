@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [268435456], []),
             B: Buffer(B_2: Pointer(float16), float16, [268435456], []),
             C: Buffer(C_2: Pointer(float16), float16, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [16384, 16384], []), B_1: B_3: Buffer(B_2, float16, [16384, 16384], []), C_1: C_3: Buffer(C_2, float16, [16384, 16384], [])} {
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 128;
  allocate(C.local: Pointer(local float16), float16, [64]), storage_scope = local;
  allocate(A.shared: Pointer(shared float16), float16, [2048]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float16), float16, [2048]), storage_scope = shared;
  allocate(A.shared.local: Pointer(local float16), float16, [8]), storage_scope = local;
  allocate(B.shared.local: Pointer(local float16), float16, [8]), storage_scope = local;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 128;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 16;
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
    for (ii.c.init: int32, 0, 4) {
      for (jj.c.init: int32, 0, 4) {
        let cse_var_1: int32 = ((ii.c.init*4) + jj.c.init)
         {
          C.local_1: Buffer(C.local, float16, [8192], [], scope="local", align=32)[cse_var_1] = 0f16
          C.local_1[(cse_var_1 + 32)] = 0f16
          C.local_1[(cse_var_1 + 16)] = 0f16
          C.local_1[(cse_var_1 + 48)] = 0f16
        }
      }
    }
    for (k.outer: int32, 0, 1024) {
      for (ax1.outer: int32, 0, 2) {
        let cse_var_2: int32 = (ax1.outer*64)
        A.shared_1: Buffer(A.shared, float16, [2048], [], scope="shared")[ramp((((threadIdx.y*128) + cse_var_2) + (threadIdx.x*4)), 1, 4)] = A[ramp((((((k.outer*262144) + (threadIdx.y*16384)) + (blockIdx.y*128)) + cse_var_2) + (threadIdx.x*4)), 1, 4)]
      }
      for (ax1.outer_1: int32, 0, 2) {
        let cse_var_3: int32 = (ax1.outer_1*64)
        B.shared_1: Buffer(B.shared, float16, [2048], [], scope="shared")[ramp((((threadIdx.y*128) + cse_var_3) + (threadIdx.x*4)), 1, 4)] = B[ramp((((((k.outer*262144) + (threadIdx.y*16384)) + (blockIdx.x*128)) + cse_var_3) + (threadIdx.x*4)), 1, 4)]
      }
      for (k.inner: int32, 0, 16) {
        for (ax1: int32, 0, 4) {
          A.shared.local_1: Buffer(A.shared.local, float16, [16], [], scope="local", align=8)[ax1] = A.shared_1[(((k.inner*128) + (threadIdx.y*4)) + ax1)]
          A.shared.local_1[(ax1 + 4)] = A.shared_1[((((k.inner*128) + (threadIdx.y*4)) + ax1) + 64)]
        }
        for (ax1_1: int32, 0, 4) {
          B.shared.local_1: Buffer(B.shared.local, float16, [16], [], scope="local", align=8)[ax1_1] = B.shared_1[(((k.inner*128) + (threadIdx.x*4)) + ax1_1)]
          B.shared.local_1[(ax1_1 + 4)] = B.shared_1[((((k.inner*128) + (threadIdx.x*4)) + ax1_1) + 64)]
        }
        for (ii.c: int32, 0, 4) {
          for (jj.c: int32, 0, 4) {
            let cse_var_9: int32 = (jj.c + 4)
            let cse_var_8: int32 = (ii.c + 4)
            let cse_var_7: int32 = ((ii.c*4) + jj.c)
            let cse_var_6: int32 = (cse_var_7 + 48)
            let cse_var_5: int32 = (cse_var_7 + 32)
            let cse_var_4: int32 = (cse_var_7 + 16)
             {
              C.local_1[cse_var_7] = (C.local_1[cse_var_7] + (A.shared.local_1[jj.c]*B.shared.local_1[ii.c]))
              C.local_1[cse_var_5] = (C.local_1[cse_var_5] + (A.shared.local_1[cse_var_9]*B.shared.local_1[ii.c]))
              C.local_1[cse_var_4] = (C.local_1[cse_var_4] + (A.shared.local_1[jj.c]*B.shared.local_1[cse_var_8]))
              C.local_1[cse_var_6] = (C.local_1[cse_var_6] + (A.shared.local_1[cse_var_9]*B.shared.local_1[cse_var_8]))
            }
          }
        }
      }
    }
    for (ii.inner.inner.inner: int32, 0, 4) {
      for (jj.inner.inner.inner: int32, 0, 4) {
        let cse_var_10: int32 = ((ii.inner.inner.inner*4) + jj.inner.inner.inner)
         {
          C[((((((blockIdx.x*2097152) + (threadIdx.x*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*4)) + jj.inner.inner.inner)] = C.local_1[cse_var_10]
          C[(((((((blockIdx.x*2097152) + (threadIdx.x*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*4)) + jj.inner.inner.inner) + 64)] = C.local_1[(cse_var_10 + 32)]
          C[(((((((blockIdx.x*2097152) + (threadIdx.x*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*4)) + jj.inner.inner.inner) + 1048576)] = C.local_1[(cse_var_10 + 16)]
          C[(((((((blockIdx.x*2097152) + (threadIdx.x*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.y*128)) + (threadIdx.y*4)) + jj.inner.inner.inner) + 1048640)] = C.local_1[(cse_var_10 + 48)]
        }
      }
    }
  }
}

