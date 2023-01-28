@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [1048576], []),
             B: Buffer(B_2: Pointer(float16), float16, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [1024, 1024], []), B_1: B_3: Buffer(B_2, float16, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], [])} {
  for (ii: int32, 0, 1024) {
    for (jj: int32, 0, 1024) {
      C[((ii*1024) + jj)] = 0f32
      for (rk: int32, 0, 1024) {
        let cse_var_2: int32 = (ii*1024)
        let cse_var_1: int32 = (cse_var_2 + jj)
        C[cse_var_1] = (C[cse_var_1] + (cast(float32, A[(cse_var_2 + rk)])*cast(float32, B[((rk*1024) + jj)])))
      }
    }
  }
}

