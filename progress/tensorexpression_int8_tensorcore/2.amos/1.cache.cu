@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.global: Pointer(global int8), int8, [268435456]), storage_scope = global;
  allocate(B.global: Pointer(global int8), int8, [268435456]), storage_scope = global;
  allocate(C.global: Pointer(global int32), int32, [268435456]), storage_scope = global {
    for (ax0: int32, 0, 16384) {
      for (ax1: int32, 0, 16384) {
        let cse_var_1: int32 = ((ax0*16384) + ax1)
        A.global_1: Buffer(A.global, int8, [268435456], [])[cse_var_1] = A[cse_var_1]
      }
    }
    for (ax0_1: int32, 0, 16384) {
      for (ax1_1: int32, 0, 16384) {
        let cse_var_2: int32 = ((ax0_1*16384) + ax1_1)
        B.global_1: Buffer(B.global, int8, [268435456], [])[cse_var_2] = B[cse_var_2]
      }
    }
    for (ii.c: int32, 0, 16384) {
      for (jj.c: int32, 0, 16384) {
        C.global_1: Buffer(C.global, int32, [268435456], [])[((ii.c*16384) + jj.c)] = 0
        for (rk: int32, 0, 16384) {
          let cse_var_4: int32 = (ii.c*16384)
          let cse_var_3: int32 = (cse_var_4 + jj.c)
          C.global_1[cse_var_3] = (C.global_1[cse_var_3] + (cast(int32, A.global_1[(cse_var_4 + rk)])*cast(int32, B.global_1[((jj.c*16384) + rk)])))
        }
      }
    }
    for (ii: int32, 0, 16384) {
      for (jj: int32, 0, 16384) {
        let cse_var_5: int32 = ((ii*16384) + jj)
        C[cse_var_5] = C.global_1[cse_var_5]
      }
    }
  }
}

