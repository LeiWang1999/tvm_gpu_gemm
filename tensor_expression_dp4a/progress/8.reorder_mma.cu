@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(C.local: Pointer(local int32), int32, [268435456]), storage_scope = local {
    attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 128;
    allocate(A.shared: Pointer(shared int8), int8, [4096]), storage_scope = shared;
    allocate(B.shared: Pointer(shared int8), int8, [4096]), storage_scope = shared;
    allocate(A.shared.local: Pointer(local int8), int8, [4096]), storage_scope = local;
    allocate(B.shared.local: Pointer(local int8), int8, [4096]), storage_scope = local;
    attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 128 {
      for (jj.c.inner.inner.outer.init: int32, 0, 64) {
        for (ii.c.inner.inner.outer.init: int32, 0, 32) {
          let cse_var_1: int32 = ((ii.c.inner.inner.outer.init*16384) + jj.c.inner.inner.outer.init)
           {
            C.local_1: Buffer(C.local, int32, [268435456], [], scope="local")[cse_var_1] = 0
            C.local_1[(cse_var_1 + 64)] = 0
            C.local_1[(cse_var_1 + 524288)] = 0
            C.local_1[(cse_var_1 + 524352)] = 0
            C.local_1[(cse_var_1 + 1048576)] = 0
            C.local_1[(cse_var_1 + 1048640)] = 0
            C.local_1[(cse_var_1 + 1572864)] = 0
            C.local_1[(cse_var_1 + 1572928)] = 0
          }
        }
      }
      for (k.outer: int32, 0, 512) {
        for (ax0: int32, 0, 128) {
          for (ax1: int32, 0, 32) {
            A.shared_1: Buffer(A.shared, int8, [4096], [], scope="shared")[((ax0*32) + ax1)] = A[((((blockIdx.y*2097152) + (ax0*16384)) + (k.outer*32)) + ax1)]
          }
        }
        for (ax0_1: int32, 0, 128) {
          for (ax1_1: int32, 0, 32) {
            B.shared_1: Buffer(B.shared, int8, [4096], [], scope="shared")[((ax0_1*32) + ax1_1)] = B[((((blockIdx.x*2097152) + (ax0_1*16384)) + (k.outer*32)) + ax1_1)]
          }
        }
        for (ax0_2: int32, 0, 64) {
          for (ax1_2: int32, 0, 32) {
            let cse_var_3: int32 = ((ax0_2*32) + ax1_2)
            let cse_var_2: int32 = (cse_var_3 + 2048)
             {
              A.shared.local_1: Buffer(A.shared.local, int8, [4194304], [], scope="local")[cse_var_3] = A.shared_1[cse_var_3]
              A.shared.local_1[cse_var_2] = A.shared_1[cse_var_2]
            }
          }
        }
        for (ax0_3: int32, 0, 32) {
          for (ax1_3: int32, 0, 32) {
            let cse_var_7: int32 = ((ax0_3*32) + ax1_3)
            let cse_var_6: int32 = (cse_var_7 + 3072)
            let cse_var_5: int32 = (cse_var_7 + 2048)
            let cse_var_4: int32 = (cse_var_7 + 1024)
             {
              B.shared.local_1: Buffer(B.shared.local, int8, [1048576], [], scope="local")[cse_var_7] = B.shared_1[cse_var_7]
              B.shared.local_1[cse_var_4] = B.shared_1[cse_var_4]
              B.shared.local_1[cse_var_5] = B.shared_1[cse_var_5]
              B.shared.local_1[cse_var_6] = B.shared_1[cse_var_6]
            }
          }
        }
        for (k.inner.inner.outer: int32, 0, 8) {
          for (jj.c.inner.inner.outer: int32, 0, 64) {
            for (ii.c.inner.inner.outer: int32, 0, 32) {
              for (k.inner.inner.inner: int32, 0, 4) {
                let cse_var_18: int32 = (k.inner.inner.outer*4)
                let cse_var_17: int32 = ((ii.c.inner.inner.outer*16384) + jj.c.inner.inner.outer)
                let cse_var_16: int32 = (cse_var_17 + 64)
                let cse_var_15: int32 = (cse_var_17 + 524352)
                let cse_var_14: int32 = (cse_var_17 + 524288)
                let cse_var_13: int32 = (cse_var_17 + 1572928)
                let cse_var_12: int32 = (cse_var_17 + 1572864)
                let cse_var_11: int32 = (cse_var_17 + 1048640)
                let cse_var_10: int32 = (cse_var_17 + 1048576)
                let cse_var_9: int32 = (((jj.c.inner.inner.outer*32) + cse_var_18) + k.inner.inner.inner)
                let cse_var_8: int32 = (((ii.c.inner.inner.outer*32) + cse_var_18) + k.inner.inner.inner)
                 {
                  C.local_1[cse_var_17] = (C.local_1[cse_var_17] + (cast(int32, A.shared.local_1[(cse_var_9 - (blockIdx.y*4096))])*cast(int32, B.shared.local_1[(cse_var_8 - (blockIdx.x*4096))])))
                  C.local_1[cse_var_16] = (C.local_1[cse_var_16] + (cast(int32, A.shared.local_1[((cse_var_9 + 2048) - (blockIdx.y*4096))])*cast(int32, B.shared.local_1[(cse_var_8 - (blockIdx.x*4096))])))
                  C.local_1[cse_var_14] = (C.local_1[cse_var_14] + (cast(int32, A.shared.local_1[(cse_var_9 - (blockIdx.y*4096))])*cast(int32, B.shared.local_1[((cse_var_8 + 1024) - (blockIdx.x*4096))])))
                  C.local_1[cse_var_15] = (C.local_1[cse_var_15] + (cast(int32, A.shared.local_1[((cse_var_9 + 2048) - (blockIdx.y*4096))])*cast(int32, B.shared.local_1[((cse_var_8 + 1024) - (blockIdx.x*4096))])))
                  C.local_1[cse_var_10] = (C.local_1[cse_var_10] + (cast(int32, A.shared.local_1[(cse_var_9 - (blockIdx.y*4096))])*cast(int32, B.shared.local_1[((cse_var_8 + 2048) - (blockIdx.x*4096))])))
                  C.local_1[cse_var_11] = (C.local_1[cse_var_11] + (cast(int32, A.shared.local_1[((cse_var_9 + 2048) - (blockIdx.y*4096))])*cast(int32, B.shared.local_1[((cse_var_8 + 2048) - (blockIdx.x*4096))])))
                  C.local_1[cse_var_12] = (C.local_1[cse_var_12] + (cast(int32, A.shared.local_1[(cse_var_9 - (blockIdx.y*4096))])*cast(int32, B.shared.local_1[((cse_var_8 + 3072) - (blockIdx.x*4096))])))
                  C.local_1[cse_var_13] = (C.local_1[cse_var_13] + (cast(int32, A.shared.local_1[((cse_var_9 + 2048) - (blockIdx.y*4096))])*cast(int32, B.shared.local_1[((cse_var_8 + 3072) - (blockIdx.x*4096))])))
                }
              }
            }
          }
        }
      }
    }
    for (ii: int32, 0, 16384) {
      for (jj: int32, 0, 16384) {
        let cse_var_19: int32 = ((ii*16384) + jj)
        C[cse_var_19] = C.local_1[cse_var_19]
      }
    }
  }
}

