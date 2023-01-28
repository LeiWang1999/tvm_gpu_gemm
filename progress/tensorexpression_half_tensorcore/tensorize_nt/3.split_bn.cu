@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [8192], []),
             B: Buffer(B_2: Pointer(float16), float16, [16384], []),
             C: Buffer(C_2: Pointer(float32), float32, [32768], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [128, 64], []), B_1: B_3: Buffer(B_2, float16, [256, 64], []), C_1: C_3: Buffer(C_2, float32, [128, 256], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [16384]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [8192]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [16384]), storage_scope = wmma.matrix_b;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [32768]), storage_scope = wmma.accumulator;
  allocate(C.wmma.accumulator.shared: Pointer(shared float32), float32, [32768]), storage_scope = shared {
    for (ax0: int32, 0, 128) {
      for (ax1: int32, 0, 64) {
        let cse_var_1: int32 = ((ax0*64) + ax1)
        A.shared_1: Buffer(A.shared, float16, [8192], [], scope="shared")[cse_var_1] = A[cse_var_1]
      }
    }
    for (ax0_1: int32, 0, 128) {
      for (ax1_1: int32, 0, 64) {
        let cse_var_2: int32 = ((ax0_1*64) + ax1_1)
        A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [8192], [], scope="wmma.matrix_a")[cse_var_2] = A.shared_1[cse_var_2]
      }
    }
    for (ax0_2: int32, 0, 256) {
      for (ax1_2: int32, 0, 64) {
        let cse_var_3: int32 = ((ax0_2*64) + ax1_2)
        A.shared_2: Buffer(A.shared, float16, [16384], [], scope="shared")[cse_var_3] = B[cse_var_3]
      }
    }
    for (ax0_3: int32, 0, 256) {
      for (ax1_3: int32, 0, 64) {
        let cse_var_4: int32 = ((ax0_3*64) + ax1_3)
        B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [16384], [], scope="wmma.matrix_b")[cse_var_4] = A.shared_2[cse_var_4]
      }
    }
    for (ii.c: int32, 0, 128) {
      for (jj.c: int32, 0, 256) {
        C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [32768], [], scope="wmma.accumulator")[((ii.c*256) + jj.c)] = 0f32
        for (rk: int32, 0, 64) {
          let cse_var_5: int32 = ((ii.c*256) + jj.c)
          C.wmma.accumulator_1[cse_var_5] = (C.wmma.accumulator_1[cse_var_5] + (cast(float32, A.shared.wmma.matrix_a_1[((ii.c*64) + rk)])*cast(float32, B.shared.wmma.matrix_b_1[((jj.c*64) + rk)])))
        }
      }
    }
    for (ax0_4: int32, 0, 128) {
      for (ax1_4: int32, 0, 256) {
        let cse_var_6: int32 = ((ax0_4*256) + ax1_4)
        C.wmma.accumulator.shared_1: Buffer(C.wmma.accumulator.shared, float32, [32768], [], scope="shared")[cse_var_6] = C.wmma.accumulator_1[cse_var_6]
      }
    }
    for (ii.outer: int32, 0, 8) {
      for (ii.inner: int32, 0, 16) {
        for (jj.outer: int32, 0, 16) {
          for (jj.inner: int32, 0, 16) {
            let cse_var_7: int32 = ((((ii.outer*4096) + (ii.inner*256)) + (jj.outer*16)) + jj.inner)
            C[cse_var_7] = C.wmma.accumulator.shared_1[cse_var_7]
          }
        }
      }
    }
  }
}

