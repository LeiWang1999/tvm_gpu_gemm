@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  for (ii: int32, 0, 16384) {
    for (jj: int32, 0, 16384) {
      C[((ii*16384) + jj)] = 0
      for (rk: int32, 0, 16384) {
        let cse_var_2: int32 = (ii*16384)
        let cse_var_1: int32 = (cse_var_2 + jj)
        C[cse_var_1] = (C[cse_var_1] + (cast(int32, A[(cse_var_2 + rk)])*cast(int32, B[((jj*16384) + rk)])))
      }
    }
  }
}

