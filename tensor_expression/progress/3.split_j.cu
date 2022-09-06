@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [268435456], []),
             B: Buffer(B_2: Pointer(float32), float32, [268435456], []),
             C: Buffer(C_2: Pointer(float32), float32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [16384, 16384], []), B_1: B_3: Buffer(B_2, float32, [16384, 16384], []), C_1: C_3: Buffer(C_2, float32, [16384, 16384], [])} {
  allocate(A.shared: Pointer(shared float32), float32, [268435456i64]), storage_scope = shared;
  allocate(A.shared.local: Pointer(local float32), float32, [268435456]), storage_scope = local;
  allocate(B.shared.local: Pointer(local float32), float32, [268435456]), storage_scope = local;
  allocate(C.local: Pointer(local float32), float32, [268435456]), storage_scope = local {
    for (ax0: int32, 0, 16384) {
      for (ax1: int32, 0, 16384) {
        let cse_var_1: int32 = ((ax0*16384) + ax1)
        A.shared_1: Buffer(A.shared, float32, [268435456], [], scope="shared")[cse_var_1] = A[cse_var_1]
      }
    }
    for (ax0_1: int32, 0, 16384) {
      for (ax1_1: int32, 0, 16384) {
        let cse_var_2: int32 = ((ax0_1*16384) + ax1_1)
        A.shared.local_1: Buffer(A.shared.local, float32, [268435456], [], scope="local")[cse_var_2] = A.shared_1[cse_var_2]
      }
    }
    for (ax0_2: int32, 0, 16384) {
      for (ax1_2: int32, 0, 16384) {
        let cse_var_3: int32 = ((ax0_2*16384) + ax1_2)
        A.shared_2: Buffer(A.shared, float32, [268435456], [], scope="shared")[cse_var_3] = B[cse_var_3]
      }
    }
    for (ax0_3: int32, 0, 16384) {
      for (ax1_3: int32, 0, 16384) {
        let cse_var_4: int32 = ((ax0_3*16384) + ax1_3)
        B.shared.local_1: Buffer(B.shared.local, float32, [268435456], [], scope="local")[cse_var_4] = A.shared_2[cse_var_4]
      }
    }
    for (ii.c: int32, 0, 16384) {
      for (jj.c: int32, 0, 16384) {
        C.local_1: Buffer(C.local, float32, [268435456], [], scope="local")[((ii.c*16384) + jj.c)] = 0f32
        for (k: int32, 0, 16384) {
          let cse_var_6: int32 = (k*16384)
          let cse_var_5: int32 = ((ii.c*16384) + jj.c)
          C.local_1[cse_var_5] = (C.local_1[cse_var_5] + (A.shared.local_1[(cse_var_6 + jj.c)]*B.shared.local_1[(cse_var_6 + ii.c)]))
        }
      }
    }
    for (ii.outer: int32, 0, 128) {
      for (ii.inner: int32, 0, 128) {
        for (jj.outer: int32, 0, 128) {
          for (jj.inner: int32, 0, 128) {
            let cse_var_7: int32 = ((((ii.outer*2097152) + (ii.inner*16384)) + (jj.outer*128)) + jj.inner)
            C[cse_var_7] = C.local_1[cse_var_7]
          }
        }
      }
    }
  }
}

