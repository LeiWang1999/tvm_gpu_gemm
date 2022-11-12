@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.shared: Pointer(shared int8), int8, [268435456i64]), storage_scope = shared;
  allocate(A.shared.local: Pointer(local int8), int8, [268435456]), storage_scope = local;
  allocate(B.shared.local: Pointer(local int8), int8, [268435456]), storage_scope = local;
  allocate(C.local: Pointer(local int32), int32, [268435456]), storage_scope = local {
    for (ax0: int32, 0, 16384) {
      for (ax1: int32, 0, 16384) {
        let cse_var_1: int32 = ((ax0*16384) + ax1)
        A.shared_1: Buffer(A.shared, int8, [268435456], [], scope="shared")[cse_var_1] = A[cse_var_1]
      }
    }
    for (ax0_1: int32, 0, 16384) {
      for (ax1_1: int32, 0, 16384) {
        let cse_var_2: int32 = ((ax0_1*16384) + ax1_1)
        A.shared.local_1: Buffer(A.shared.local, int8, [268435456], [], scope="local")[cse_var_2] = A.shared_1[cse_var_2]
      }
    }
    for (ax0_2: int32, 0, 16384) {
      for (ax1_2: int32, 0, 16384) {
        let cse_var_3: int32 = ((ax0_2*16384) + ax1_2)
        A.shared_2: Buffer(A.shared, int8, [268435456], [], scope="shared")[cse_var_3] = B[cse_var_3]
      }
    }
    for (ax0_3: int32, 0, 16384) {
      for (ax1_3: int32, 0, 16384) {
        let cse_var_4: int32 = ((ax0_3*16384) + ax1_3)
        B.shared.local_1: Buffer(B.shared.local, int8, [268435456], [], scope="local")[cse_var_4] = A.shared_2[cse_var_4]
      }
    }
    for (jj.c.outer: int32, 0, 128) {
      for (ii.c.outer: int32, 0, 128) {
        for (ii.c.inner.init: int32, 0, 128) {
          for (jj.c.inner.init: int32, 0, 128) {
            C.local_1: Buffer(C.local, int32, [268435456], [], scope="local")[((((ii.c.outer*2097152) + (ii.c.inner.init*16384)) + (jj.c.outer*128)) + jj.c.inner.init)] = 0
          }
        }
        for (k.outer: int32, 0, 512) {
          for (ii.c.inner: int32, 0, 128) {
            for (jj.c.inner: int32, 0, 128) {
              for (k.inner: int32, 0, 32) {
                let cse_var_7: int32 = ((ii.c.outer*2097152) + (ii.c.inner*16384))
                let cse_var_6: int32 = (k.outer*32)
                let cse_var_5: int32 = ((cse_var_7 + (jj.c.outer*128)) + jj.c.inner)
                C.local_1[cse_var_5] = (C.local_1[cse_var_5] + (cast(int32, A.shared.local_1[((((jj.c.outer*2097152) + (jj.c.inner*16384)) + cse_var_6) + k.inner)])*cast(int32, B.shared.local_1[((cse_var_7 + cse_var_6) + k.inner)])))
              }
            }
          }
        }
      }
    }
    for (ii: int32, 0, 16384) {
      for (jj: int32, 0, 16384) {
        let cse_var_8: int32 = ((ii*16384) + jj)
        C[cse_var_8] = C.local_1[cse_var_8]
      }
    }
  }
}

