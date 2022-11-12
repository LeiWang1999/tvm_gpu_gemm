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
    allocate(A.shared.local: Pointer(local int8), int8, [2048]), storage_scope = local;
    allocate(B.shared.local: Pointer(local int8), int8, [1024]), storage_scope = local;
    attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 128 {
      for (jj.c.inner.outer.init: int32, 0, 2) {
        for (ii.c.inner.outer.init: int32, 0, 4) {
          for (jj.c.inner.inner.init: int32, 0, 64) {
            for (ii.c.inner.inner.init: int32, 0, 32) {
              C.local_1: Buffer(C.local, int32, [268435456], [], scope="local")[((((ii.c.inner.outer.init*524288) + (ii.c.inner.inner.init*16384)) + (jj.c.inner.outer.init*64)) + jj.c.inner.inner.init)] = 0
            }
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
        for (jj.c.inner.outer: int32, 0, 2) {
          for (ii.c.inner.outer: int32, 0, 4) {
            for (ax0_2: int32, 0, 64) {
              for (ax1_2: int32, 0, 32) {
                let cse_var_1: int32 = (ax0_2*32)
                A.shared.local_1: Buffer(A.shared.local, int8, [2048], [], scope="local")[(cse_var_1 + ax1_2)] = A.shared_1[(((jj.c.inner.outer*2048) + cse_var_1) + ax1_2)]
              }
            }
            for (ax0_3: int32, 0, 32) {
              for (ax1_3: int32, 0, 32) {
                let cse_var_2: int32 = (ax0_3*32)
                B.shared.local_1: Buffer(B.shared.local, int8, [1024], [], scope="local")[(cse_var_2 + ax1_3)] = B.shared_1[(((ii.c.inner.outer*1024) + cse_var_2) + ax1_3)]
              }
            }
            for (jj.c.inner.inner: int32, 0, 64) {
              for (ii.c.inner.inner: int32, 0, 32) {
                for (k.inner.inner: int32, 0, 32) {
                  let cse_var_3: int32 = ((((ii.c.inner.outer*524288) + (ii.c.inner.inner*16384)) + (jj.c.inner.outer*64)) + jj.c.inner.inner)
                  C.local_1[cse_var_3] = (C.local_1[cse_var_3] + (cast(int32, A.shared.local_1[(((jj.c.inner.inner*32) + k.inner.inner) - (blockIdx.y*4096))])*cast(int32, B.shared.local_1[(((ii.c.inner.inner*32) + k.inner.inner) - (blockIdx.x*4096))])))
                }
              }
            }
          }
        }
      }
    }
    for (ii: int32, 0, 16384) {
      for (jj: int32, 0, 16384) {
        let cse_var_4: int32 = ((ii*16384) + jj)
        C[cse_var_4] = C.local_1[cse_var_4]
      }
    }
  }
}

