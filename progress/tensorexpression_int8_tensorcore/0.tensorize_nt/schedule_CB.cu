@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.global: Pointer(global int8), int8, [268435456i64]), storage_scope = global;
  allocate(A.global.shared: Pointer(shared int8), int8, [268435456i64]), storage_scope = shared;
  allocate(A.global.shared.wmma.matrix_a: Pointer(wmma.matrix_a int8), int8, [268435456]), storage_scope = wmma.matrix_a;
  allocate(B.global.shared.wmma.matrix_b: Pointer(wmma.matrix_b int8), int8, [268435456]), storage_scope = wmma.matrix_b;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator int32), int32, [268435456]), storage_scope = wmma.accumulator;
  allocate(C.wmma.accumulator.global: Pointer(global int32), int32, [268435456]), storage_scope = global {
    for (ax0: int32, 0, 16384) {
      for (ax1: int32, 0, 16384) {
        let cse_var_1: int32 = ((ax0*16384) + ax1)
        A.global_1: Buffer(A.global, int8, [268435456], [])[cse_var_1] = A[cse_var_1]
      }
    }
    for (ax0_1: int32, 0, 16384) {
      for (ax1_1: int32, 0, 16384) {
        let cse_var_2: int32 = ((ax0_1*16384) + ax1_1)
        A.global.shared_1: Buffer(A.global.shared, int8, [268435456], [], scope="shared")[cse_var_2] = A.global_1[cse_var_2]
      }
    }
    for (ax0_2: int32, 0, 16384) {
      for (ax1_2: int32, 0, 16384) {
        let cse_var_3: int32 = ((ax0_2*16384) + ax1_2)
        A.global.shared.wmma.matrix_a_1: Buffer(A.global.shared.wmma.matrix_a, int8, [268435456], [], scope="wmma.matrix_a")[cse_var_3] = A.global.shared_1[cse_var_3]
      }
    }
    for (ax0_3: int32, 0, 16384) {
      for (ax1_3: int32, 0, 16384) {
        let cse_var_4: int32 = ((ax0_3*16384) + ax1_3)
        A.global_2: Buffer(A.global, int8, [268435456], [])[cse_var_4] = B[cse_var_4]
      }
    }
    for (ax0_4: int32, 0, 16384) {
      for (ax1_4: int32, 0, 16384) {
        let cse_var_5: int32 = ((ax0_4*16384) + ax1_4)
        A.global.shared_2: Buffer(A.global.shared, int8, [268435456], [], scope="shared")[cse_var_5] = A.global_2[cse_var_5]
      }
    }
    for (ax0_5: int32, 0, 16384) {
      for (ax1_5: int32, 0, 16384) {
        let cse_var_6: int32 = ((ax0_5*16384) + ax1_5)
        B.global.shared.wmma.matrix_b_1: Buffer(B.global.shared.wmma.matrix_b, int8, [268435456], [], scope="wmma.matrix_b")[cse_var_6] = A.global.shared_2[cse_var_6]
      }
    }
    for (ii.c: int32, 0, 16384) {
      for (jj.c: int32, 0, 16384) {
        C.wmma.accumulator_1: Buffer(C.wmma.accumulator, int32, [268435456], [], scope="wmma.accumulator")[((ii.c*16384) + jj.c)] = 0
        for (rk: int32, 0, 16384) {
          let cse_var_8: int32 = (ii.c*16384)
          let cse_var_7: int32 = (cse_var_8 + jj.c)
          C.wmma.accumulator_1[cse_var_7] = (C.wmma.accumulator_1[cse_var_7] + (cast(int32, A.global.shared.wmma.matrix_a_1[(cse_var_8 + rk)])*cast(int32, B.global.shared.wmma.matrix_b_1[((jj.c*16384) + rk)])))
        }
      }
    }
    for (ax0_6: int32, 0, 16384) {
      for (ax1_6: int32, 0, 16384) {
        let cse_var_9: int32 = ((ax0_6*16384) + ax1_6)
        C.wmma.accumulator.global_1: Buffer(C.wmma.accumulator.global, int32, [268435456], [])[cse_var_9] = C.wmma.accumulator_1[cse_var_9]
      }
    }
    for (ii: int32, 0, 16384) {
      for (jj: int32, 0, 16384) {
        let cse_var_10: int32 = ((ii*16384) + jj)
        C[cse_var_10] = C.wmma.accumulator.global_1[cse_var_10]
      }
    }
  }
}

