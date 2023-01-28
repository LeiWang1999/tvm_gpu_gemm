@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [1024, 1024, 16, 16], []), B_1: B_3: Buffer(B_2, int8, [1024, 1024, 16, 16], []), C_1: C_3: Buffer(C_2, int32, [1024, 1024, 16, 16], [])} {
  allocate(B.shared: Pointer(shared int8), int8, [4194304]), storage_scope = shared {
    for (ax0: int32, 0, 16) {
      for (ax1: int32, 0, 1024) {
        for (ax2: int32, 0, 16) {
          for (ax3: int32, 0, 16) {
            let cse_var_3: int32 = (ax0*262144)
            let cse_var_2: int32 = (ax1*256)
            let cse_var_1: int32 = (ax2*16)
            B.shared_1: Buffer(B.shared, int8, [4194304], [], scope="shared")[(((cse_var_3 + cse_var_2) + cse_var_1) + ax3)] = B[(((((blockIdx.y: int32*4194304) + cse_var_3) + cse_var_2) + cse_var_1) + ax3)]
          }
        }
      }
    }
    attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 256;
    allocate(C.wmma.accumulator: Pointer(wmma.accumulator int32), int32, [4096]), storage_scope = wmma.accumulator;
    allocate(A.shared: Pointer(shared int8), int8, [4096]), storage_scope = shared;
    allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a int8), int8, [512]), storage_scope = wmma.matrix_a;
    allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b int8), int8, [2048]), storage_scope = wmma.matrix_b;
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 64;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
    attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2 {
      for (i.c.init: int32, 0, 2) {
        for (j.c.init: int32, 0, 8) {
          for (ii.c.init: int32, 0, 16) {
            for (jj.c.init: int32, 0, 16) {
              C.wmma.accumulator_1: Buffer(C.wmma.accumulator, int32, [4096], [], scope="wmma.accumulator")[((((i.c.init*2048) + (j.c.init*256)) + (ii.c.init*16)) + jj.c.init)] = 0
            }
          }
        }
      }
      for (k1.outer: int32, 0, 256) {
        for (ax0_1: int32, 0, 4) {
          for (ax1_1: int32, 0, 4) {
            for (ax2_1: int32, 0, 16) {
              for (ax3_1: int32, 0, 16) {
                let cse_var_5: int32 = (ax1_1*256)
                let cse_var_4: int32 = (ax2_1*16)
                A.shared_1: Buffer(A.shared, int8, [4096], [], scope="shared")[((((ax0_1*1024) + cse_var_5) + cse_var_4) + ax3_1)] = A[((((((blockIdx.x*1048576) + (ax0_1*262144)) + (k1.outer*1024)) + cse_var_5) + cse_var_4) + ax3_1)]
              }
            }
          }
        }
        for (k1.inner: int32, 0, 4) {
          for (ax0_2: int32, 0, 2) {
            for (ax2_2: int32, 0, 16) {
              for (ax3_2: int32, 0, 16) {
                let cse_var_6: int32 = (ax2_2*16)
                A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, int8, [512], [], scope="wmma.matrix_a")[(((ax0_2*256) + cse_var_6) + ax3_2)] = A.shared_1[(((((threadIdx.y*2048) + (ax0_2*1024)) + (k1.inner*256)) + cse_var_6) + ax3_2)]
              }
            }
          }
          for (ax0_3: int32, 0, 8) {
            for (ax2_3: int32, 0, 16) {
              for (ax3_3: int32, 0, 16) {
                let cse_var_7: int32 = (ax2_3*16)
                B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, int8, [2048], [], scope="wmma.matrix_b")[(((ax0_3*256) + cse_var_7) + ax3_3)] = B.shared_1[((((((threadIdx.z*2097152) + (ax0_3*262144)) + (k1.outer*1024)) + (k1.inner*256)) + cse_var_7) + ax3_3)]
              }
            }
          }
          for (i.c: int32, 0, 2) {
            for (j.c: int32, 0, 8) {
              for (ii.c: int32, 0, 16) {
                for (jj.c: int32, 0, 16) {
                  for (k2: int32, 0, 16) {
                    let cse_var_10: int32 = (j.c*256)
                    let cse_var_9: int32 = (ii.c*16)
                    let cse_var_8: int32 = ((((i.c*2048) + cse_var_10) + cse_var_9) + jj.c)
                    C.wmma.accumulator_1[cse_var_8] = (C.wmma.accumulator_1[cse_var_8] + (cast(int32, A.shared.wmma.matrix_a_1[(((i.c*256) + cse_var_9) + k2)])*cast(int32, B.shared.wmma.matrix_b_1[((cse_var_10 + (jj.c*16)) + k2)])))
                  }
                }
              }
            }
          }
        }
      }
      for (i.inner: int32, 0, 2) {
        for (j.inner: int32, 0, 8) {
          for (ii: int32, 0, 16) {
            for (jj: int32, 0, 16) {
              let cse_var_12: int32 = (j.inner*256)
              let cse_var_11: int32 = (ii*16)
              C[((((((((blockIdx.x*1048576) + (threadIdx.y*524288)) + (i.inner*262144)) + (blockIdx.y*4096)) + (threadIdx.z*2048)) + cse_var_12) + cse_var_11) + jj)] = C.wmma.accumulator_1[((((i.inner*2048) + cse_var_12) + cse_var_11) + jj)]
            }
          }
        }
      }
    }
  }
}

