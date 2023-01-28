@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main", "from_legacy_te_schedule": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.shared: Pointer(shared int8), int8, [268435456]), storage_scope = shared;
  allocate(B.shared: Pointer(shared int8), int8, [268435456]), storage_scope = shared;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator int32), int32, [268435456]), storage_scope = wmma.accumulator {
    for (axis0: int32, 0, 1024) {
      for (axis1: int32, 0, 1024) {
        for (axis2: int32, 0, 16) {
          for (axis3: int32, 0, 16) {
            let cse_var_1: int32 = (axis0*262144)
            A.shared_1: Buffer(A.shared, int8, [268435456], [], scope="shared")[(((cse_var_1 + (axis1*256)) + (axis2*16)) + axis3)] = A[(((cse_var_1 + (axis2*16384)) + (axis1*16)) + axis3)]
          }
        }
      }
    }
    for (axis0_1: int32, 0, 1024) {
      for (axis1_1: int32, 0, 1024) {
        for (axis2_1: int32, 0, 16) {
          for (axis3_1: int32, 0, 16) {
            let cse_var_2: int32 = (axis0_1*262144)
            B.shared_1: Buffer(B.shared, int8, [268435456], [], scope="shared")[(((cse_var_2 + (axis1_1*256)) + (axis2_1*16)) + axis3_1)] = B[(((cse_var_2 + (axis2_1*16384)) + (axis1_1*16)) + axis3_1)]
          }
        }
      }
    }
    for (axis0_2: int32, 0, 1024) {
      for (axis1_2: int32, 0, 1024) {
        for (axis2_2: int32, 0, 16) {
          for (axis3_2: int32, 0, 16) {
            C.wmma.accumulator_1: Buffer(C.wmma.accumulator, int32, [268435456], [], scope="wmma.accumulator")[((((axis0_2*262144) + (axis1_2*256)) + (axis2_2*16)) + axis3_2)] = 0
            for (rk: int32, 0, 16384) {
              let cse_var_7: int32 = (axis0_2*262144)
              let cse_var_6: int32 = (axis2_2*16)
              let cse_var_5: int32 = floormod(rk, 16)
              let cse_var_4: int32 = (floordiv(rk, 16)*256)
              let cse_var_3: int32 = (((cse_var_7 + (axis1_2*256)) + cse_var_6) + axis3_2)
              C.wmma.accumulator_1[cse_var_3] = (C.wmma.accumulator_1[cse_var_3] + (cast(int32, A.shared_1[(((cse_var_7 + cse_var_4) + cse_var_6) + cse_var_5)])*cast(int32, B.shared_1[((((axis1_2*262144) + cse_var_4) + (axis3_2*16)) + cse_var_5)])))
            }
          }
        }
      }
    }
    for (ii: int32, 0, 16384) {
      for (jj: int32, 0, 16384) {
        C[((ii*16384) + jj)] = C.wmma.accumulator_1[((((floordiv(ii, 16)*262144) + (floordiv(jj, 16)*256)) + (floormod(ii, 16)*16)) + floormod(jj, 16))]
      }
    }
  }
}

