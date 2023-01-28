@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main", "from_legacy_te_schedule": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.global: Pointer(global int8), int8, [268435456]), storage_scope = global;
  allocate(B.global: Pointer(global int8), int8, [268435456]), storage_scope = global;
  allocate(B.global.shared: Pointer(shared int8), int8, [268435456]), storage_scope = shared;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator int32), int32, [268435456]), storage_scope = wmma.accumulator;
  allocate(A.global.shared: Pointer(shared int8), int8, [1048576]), storage_scope = shared;
  allocate(A.global.shared.wmma.matrix_a: Pointer(wmma.matrix_a int8), int8, [262144]), storage_scope = wmma.matrix_a;
  allocate(B.global.shared.wmma.matrix_b: Pointer(wmma.matrix_b int8), int8, [262144]), storage_scope = wmma.matrix_b;
  allocate(C.wmma.accumulator.global: Pointer(global int32), int32, [268435456]), storage_scope = global {
    attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1024;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 32;
    attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    for (axis2.axis3.fused.outer: int32, 0, 16) {
      A.global_1: Buffer(A.global, int8, [268435456], [])[ramp(((((blockIdx.x*262144) + (threadIdx.y*8192)) + (threadIdx.x*256)) + (axis2.axis3.fused.outer*16)), 1, 16)] = A[ramp(((((blockIdx.x*262144) + (axis2.axis3.fused.outer*16384)) + (threadIdx.y*512)) + (threadIdx.x*16)), 1, 16)]
    }
    attr [IterVar(blockIdx.x_1: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1024;
    attr [IterVar(threadIdx.y_1: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 32;
    attr [IterVar(threadIdx.x_1: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    for (axis2.axis3.fused.outer_1: int32, 0, 16) {
      B.global_1: Buffer(B.global, int8, [268435456], [])[ramp(((((blockIdx.x_1*262144) + (threadIdx.y_1*8192)) + (threadIdx.x_1*256)) + (axis2.axis3.fused.outer_1*16)), 1, 16)] = B[ramp(((((blockIdx.x_1*262144) + (axis2.axis3.fused.outer_1*16384)) + (threadIdx.y_1*512)) + (threadIdx.x_1*16)), 1, 16)]
    }
    for (ax0: int32, 0, 16384) {
      for (ax1: int32, 0, 16384) {
        B.global.shared_1: Buffer(B.global.shared, int8, [268435456], [], scope="shared")[((ax0*16384) + ax1)] = B.global_1[((((floordiv(ax0, 16)*262144) + (floordiv(ax1, 16)*256)) + (floormod(ax0, 16)*16)) + floormod(ax1, 16))]
      }
    }
    for (axis0.init: int32, 0, 1024) {
      for (axis1.init: int32, 0, 1024) {
        for (axis2.init: int32, 0, 16) {
          for (axis3.init: int32, 0, 16) {
            C.wmma.accumulator_1: Buffer(C.wmma.accumulator, int32, [268435456], [], scope="wmma.accumulator")[((((axis0.init*262144) + (axis1.init*256)) + (axis2.init*16)) + axis3.init)] = 0
          }
        }
      }
    }
    for (rk.outer.outer: int32, 0, 256) {
      for (axis0: int32, 0, 1024) {
        for (axis1.idx: int32, 0, 4) {
          for (axis2: int32, 0, 16) {
            for (axis3: int32, 0, 16) {
              let cse_var_2: int32 = (axis1.idx*256)
              let cse_var_1: int32 = (axis2*16)
              A.global.shared_1: Buffer(A.global.shared, int8, [1048576], [], scope="shared")[((((axis0*1024) + cse_var_2) + cse_var_1) + axis3)] = A.global_1[(((((axis0*262144) + (rk.outer.outer*1024)) + cse_var_2) + cse_var_1) + axis3)]
            }
          }
        }
      }
      for (rk.outer.inner: int32, 0, 4) {
        for (axis0_1: int32, 0, 1024) {
          for (axis2_1: int32, 0, 16) {
            for (axis3_1: int32, 0, 16) {
              let cse_var_3: int32 = (axis2_1*16)
              A.global.shared.wmma.matrix_a_1: Buffer(A.global.shared.wmma.matrix_a, int8, [262144], [], scope="wmma.matrix_a")[(((axis0_1*256) + cse_var_3) + axis3_1)] = A.global.shared_1[((((axis0_1*1024) + (rk.outer.inner*256)) + cse_var_3) + axis3_1)]
            }
          }
        }
        for (axis0_2: int32, 0, 1024) {
          for (axis2_2: int32, 0, 16) {
            for (axis3_2: int32, 0, 16) {
              B.global.shared.wmma.matrix_b_1: Buffer(B.global.shared.wmma.matrix_b, int8, [262144], [], scope="wmma.matrix_b")[(((axis0_2*256) + (axis2_2*16)) + axis3_2)] = B.global.shared_1[(((((axis0_2*262144) + (axis2_2*16384)) + (rk.outer.outer*64)) + (rk.outer.inner*16)) + axis3_2)]
            }
          }
        }
        for (axis0_3: int32, 0, 1024) {
          for (axis1: int32, 0, 1024) {
            for (axis2_3: int32, 0, 16) {
              for (axis3_3: int32, 0, 16) {
                for (rk.inner: int32, 0, 16) {
                  let cse_var_6: int32 = (axis1*256)
                  let cse_var_5: int32 = (axis2_3*16)
                  let cse_var_4: int32 = ((((axis0_3*262144) + cse_var_6) + cse_var_5) + axis3_3)
                  C.wmma.accumulator_1[cse_var_4] = (C.wmma.accumulator_1[cse_var_4] + (cast(int32, A.global.shared.wmma.matrix_a_1[(((axis0_3*256) + cse_var_5) + rk.inner)])*cast(int32, B.global.shared.wmma.matrix_b_1[((cse_var_6 + (axis3_3*16)) + rk.inner)])))
                }
              }
            }
          }
        }
      }
    }
    for (ax0_1: int32, 0, 16384) {
      for (ax1_1: int32, 0, 16384) {
        C.wmma.accumulator.global_1: Buffer(C.wmma.accumulator.global, int32, [268435456], [])[((ax0_1*16384) + ax1_1)] = C.wmma.accumulator_1[((((floordiv(ax0_1, 16)*262144) + (floordiv(ax1_1, 16)*256)) + (floormod(ax0_1, 16)*16)) + floormod(ax1_1, 16))]
      }
    }
    for (ii: int32, 0, 16384) {
      for (jj: int32, 0, 16384) {
        let cse_var_7: int32 = ((ii*16384) + jj)
        C[cse_var_7] = C.wmma.accumulator.global_1[cse_var_7]
      }
    }
  }
}

