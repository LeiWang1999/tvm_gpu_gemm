@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [4096], []),
             B: Buffer(B_2: Pointer(float16), float16, [4096], []),
             C: Buffer(C_2: Pointer(float32), float32, [4096], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [64, 64], []), B_1: B_3: Buffer(B_2, float16, [64, 64], []), C_1: C_3: Buffer(C_2, float32, [64, 64], [])} {
  for (ii: int32, 0, 64) {
    for (jj: int32, 0, 64) {
      C[((ii*64) + jj)] = 0f32
      for (rk: int32, 0, 64) {
        let cse_var_2: int32 = (ii*64)
        let cse_var_1: int32 = (cse_var_2 + jj)
        C[cse_var_1] = (C[cse_var_1] + (cast(float32, A[(cse_var_2 + rk)])*cast(float32, B[((jj*64) + rk)])))
      }
    }
  }
}

