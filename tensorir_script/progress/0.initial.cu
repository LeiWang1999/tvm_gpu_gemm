@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [268435456], []),
             B: Buffer(B_2: Pointer(float16), float16, [268435456], []),
             C: Buffer(C_2: Pointer(float16), float16, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [16384, 16384], []), B_1: B_3: Buffer(B_2, float16, [16384, 16384], []), C_1: C_3: Buffer(C_2, float16, [16384, 16384], [])} {
  for (ii: int32, 0, 16384) {
    for (jj: int32, 0, 16384) {
      C[((ii*16384) + jj)] = 0f16
      for (k: int32, 0, 16384) {
        let cse_var_2: int32 = (k*16384)
        let cse_var_1: int32 = ((ii*16384) + jj)
        C[cse_var_1] = (C[cse_var_1] + (A[(cse_var_2 + jj)]*B[(cse_var_2 + ii)]))
      }
    }
  }
}

