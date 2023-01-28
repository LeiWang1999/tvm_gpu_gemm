@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [268435456], []),
             B: Buffer(B_2: Pointer(float16), float16, [268435456], []),
             C: Buffer(C_2: Pointer(float32), float32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [1024, 1024, 16, 16], []), B_1: B_3: Buffer(B_2, float16, [1024, 1024, 16, 16], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024, 16, 16], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [1048576]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [524288]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [1048576]), storage_scope = wmma.matrix_b;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [2048]), storage_scope = wmma.accumulator {
    for (ax0: int32, 0, 2) {
      for (ax1: int32, 0, 1024) {
        for (ax2: int32, 0, 16) {
          for (ax3: int32, 0, 16) {
            let cse_var_3: int32 = (ax0*262144)
            let cse_var_2: int32 = (ax1*256)
            let cse_var_1: int32 = (ax2*16)
            A.shared_1: Buffer(A.shared, float16, [524288], [], scope="shared")[(((cse_var_3 + cse_var_2) + cse_var_1) + ax3)] = A[((((((blockIdx.x: int32*1048576) + (threadIdx.y: int32*524288)) + cse_var_3) + cse_var_2) + cse_var_1) + ax3)]
          }
        }
      }
    }
    for (ax0_1: int32, 0, 2) {
      for (ax1_1: int32, 0, 1024) {
        for (ax2_1: int32, 0, 16) {
          for (ax3_1: int32, 0, 16) {
            let cse_var_4: int32 = ((((ax0_1*262144) + (ax1_1*256)) + (ax2_1*16)) + ax3_1)
            A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [524288], [], scope="wmma.matrix_a")[cse_var_4] = A.shared_1[cse_var_4]
          }
        }
      }
    }
    for (ax0_2: int32, 0, 4) {
      for (ax1_2: int32, 0, 1024) {
        for (ax2_2: int32, 0, 16) {
          for (ax3_2: int32, 0, 16) {
            let cse_var_7: int32 = (ax0_2*262144)
            let cse_var_6: int32 = (ax1_2*256)
            let cse_var_5: int32 = (ax2_2*16)
            A.shared_2: Buffer(A.shared, float16, [1048576], [], scope="shared")[(((cse_var_7 + cse_var_6) + cse_var_5) + ax3_2)] = B[((((((blockIdx.y: int32*2097152) + (threadIdx.z: int32*1048576)) + cse_var_7) + cse_var_6) + cse_var_5) + ax3_2)]
          }
        }
      }
    }
    for (ax0_3: int32, 0, 4) {
      for (ax1_3: int32, 0, 1024) {
        for (ax2_3: int32, 0, 16) {
          for (ax3_3: int32, 0, 16) {
            let cse_var_8: int32 = ((((ax0_3*262144) + (ax1_3*256)) + (ax2_3*16)) + ax3_3)
            B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [1048576], [], scope="wmma.matrix_b")[cse_var_8] = A.shared_2[cse_var_8]
          }
        }
      }
    }
    for (i.c: int32, 0, 2) {
      for (j.c: int32, 0, 4) {
        for (ii.c: int32, 0, 16) {
          for (jj.c: int32, 0, 16) {
            C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [2048], [], scope="wmma.accumulator")[((((i.c*1024) + (j.c*256)) + (ii.c*16)) + jj.c)] = 0f32
            for (k1: int32, 0, 1024) {
              for (k2: int32, 0, 16) {
                let cse_var_11: int32 = (ii.c*16)
                let cse_var_10: int32 = (k1*256)
                let cse_var_9: int32 = ((((i.c*1024) + (j.c*256)) + cse_var_11) + jj.c)
                C.wmma.accumulator_1[cse_var_9] = (C.wmma.accumulator_1[cse_var_9] + (cast(float32, A.shared.wmma.matrix_a_1[((((i.c*262144) + cse_var_10) + cse_var_11) + k2)])*cast(float32, B.shared.wmma.matrix_b_1[((((j.c*262144) + cse_var_10) + (jj.c*16)) + k2)])))
              }
            }
          }
        }
      }
    }
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 256;
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 128;
    attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
    attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2;
    for (i.inner: int32, 0, 2) {
      for (j.inner: int32, 0, 4) {
        for (ii: int32, 0, 16) {
          for (jj: int32, 0, 16) {
            let cse_var_13: int32 = (j.inner*256)
            let cse_var_12: int32 = (ii*16)
            C[((((((((blockIdx.x*1048576) + (threadIdx.y*524288)) + (i.inner*262144)) + (blockIdx.y*2048)) + (threadIdx.z*1024)) + cse_var_13) + cse_var_12) + jj)] = C.wmma.accumulator_1[((((i.inner*1024) + cse_var_13) + cse_var_12) + jj)]
          }
        }
      }
    }
  }
}

