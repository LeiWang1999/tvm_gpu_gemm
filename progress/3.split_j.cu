@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [268435456], []),
             B: Buffer(B_2: Pointer(float32), float32, [268435456], []),
             C: Buffer(C_2: Pointer(float32), float32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [16384, 16384], []), B_1: B_3: Buffer(B_2, float32, [16384, 16384], []), C_1: C_3: Buffer(C_2, float32, [16384, 16384], [])} {
  allocate(C.local: Pointer(local float32), float32, [268435456]), storage_scope = local {
    for (ii.c: int32, 0, 16384) {
      for (jj.c: int32, 0, 16384) {
        C.local_1: Buffer(C.local, float32, [268435456], [], scope="local")[((ii.c*16384) + jj.c)] = 0f32
        for (k: int32, 0, 16384) {
          let cse_var_2: int32 = (k*16384)
          let cse_var_1: int32 = ((ii.c*16384) + jj.c)
          C.local_1[cse_var_1] = (C.local_1[cse_var_1] + (A[(cse_var_2 + jj.c)]*B[(cse_var_2 + ii.c)]))
        }
      }
    }
    for (ii.outer: int32, 0, 512) {
      for (ii.inner: int32, 0, 32) {
        for (jj.outer: int32, 0, 512) {
          for (jj.inner: int32, 0, 32) {
            let cse_var_3: int32 = ((((ii.outer*524288) + (ii.inner*16384)) + (jj.outer*32)) + jj.inner)
            C[cse_var_3] = C.local_1[cse_var_3]
          }
        }
      }
    }
  }
}

