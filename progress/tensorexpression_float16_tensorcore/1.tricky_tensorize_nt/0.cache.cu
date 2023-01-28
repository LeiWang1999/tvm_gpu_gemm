@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [268435456], []),
             B: Buffer(B_2: Pointer(float16), float16, [268435456], []),
             C: Buffer(C_2: Pointer(float32), float32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [1024, 1024, 16, 16], []), B_1: B_3: Buffer(B_2, float16, [1024, 1024, 16, 16], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024, 16, 16], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [268435456]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [268435456]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [268435456]), storage_scope = wmma.matrix_b;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [268435456]), storage_scope = wmma.accumulator {
    for (ax0: int32, 0, 1024) {
      for (ax1: int32, 0, 1024) {
        for (ax2: int32, 0, 16) {
          for (ax3: int32, 0, 16) {
            let cse_var_1: int32 = ((((ax0*262144) + (ax1*256)) + (ax2*16)) + ax3)
            A.shared_1: Buffer(A.shared, float16, [268435456], [], scope="shared")[cse_var_1] = A[cse_var_1]
          }
        }
      }
    }
    for (ax0_1: int32, 0, 1024) {
      for (ax1_1: int32, 0, 1024) {
        for (ax2_1: int32, 0, 16) {
          for (ax3_1: int32, 0, 16) {
            let cse_var_2: int32 = ((((ax0_1*262144) + (ax1_1*256)) + (ax2_1*16)) + ax3_1)
            A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [268435456], [], scope="wmma.matrix_a")[cse_var_2] = A.shared_1[cse_var_2]
          }
        }
      }
    }
    for (ax0_2: int32, 0, 1024) {
      for (ax1_2: int32, 0, 1024) {
        for (ax2_2: int32, 0, 16) {
          for (ax3_2: int32, 0, 16) {
            let cse_var_3: int32 = ((((ax0_2*262144) + (ax1_2*256)) + (ax2_2*16)) + ax3_2)
            A.shared_2: Buffer(A.shared, float16, [268435456], [], scope="shared")[cse_var_3] = B[cse_var_3]
          }
        }
      }
    }
    for (ax0_3: int32, 0, 1024) {
      for (ax1_3: int32, 0, 1024) {
        for (ax2_3: int32, 0, 16) {
          for (ax3_3: int32, 0, 16) {
            let cse_var_4: int32 = ((((ax0_3*262144) + (ax1_3*256)) + (ax2_3*16)) + ax3_3)
            B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [268435456], [], scope="wmma.matrix_b")[cse_var_4] = A.shared_2[cse_var_4]
          }
        }
      }
    }
    for (i.c: int32, 0, 1024) {
      for (j.c: int32, 0, 1024) {
        for (ii.c: int32, 0, 16) {
          for (jj.c: int32, 0, 16) {
            C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [268435456], [], scope="wmma.accumulator")[((((i.c*262144) + (j.c*256)) + (ii.c*16)) + jj.c)] = 0f32
            for (k1: int32, 0, 1024) {
              for (k2: int32, 0, 16) {
                let cse_var_8: int32 = (i.c*262144)
                let cse_var_7: int32 = (ii.c*16)
                let cse_var_6: int32 = (k1*256)
                let cse_var_5: int32 = (((cse_var_8 + (j.c*256)) + cse_var_7) + jj.c)
                C.wmma.accumulator_1[cse_var_5] = (C.wmma.accumulator_1[cse_var_5] + (cast(float32, A.shared.wmma.matrix_a_1[(((cse_var_8 + cse_var_6) + cse_var_7) + k2)])*cast(float32, B.shared.wmma.matrix_b_1[((((j.c*262144) + cse_var_6) + (jj.c*16)) + k2)])))
              }
            }
          }
        }
      }
    }
    for (i: int32, 0, 1024) {
      for (j: int32, 0, 1024) {
        for (ii: int32, 0, 16) {
          for (jj: int32, 0, 16) {
            let cse_var_9: int32 = ((((i*262144) + (j*256)) + (ii*16)) + jj)
            C[cse_var_9] = C.wmma.accumulator_1[cse_var_9]
          }
        }
      }
    }
  }
}

