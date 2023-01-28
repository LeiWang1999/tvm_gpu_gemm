@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main", "from_legacy_te_schedule": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.global: Pointer(global int8), int8, [268435456i64]), storage_scope = global;
  allocate(A.global.shared: Pointer(shared int8), int8, [268435456i64]), storage_scope = shared;
  allocate(A.global.shared.local: Pointer(local int8), int8, [268435456]), storage_scope = local;
  allocate(B.global.shared.local: Pointer(local int8), int8, [268435456]), storage_scope = local;
  allocate(C.local: Pointer(local int32), int32, [268435456]), storage_scope = local;
  allocate(C.local.global: Pointer(global int32), int32, [268435456]), storage_scope = global {
    attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1024;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 32;
    attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    for (axis2.axis3.fused.outer: int32, 0, 16) {
      A.global_1: Buffer(A.global, int8, [268435456], [])[ramp(((((blockIdx.x*262144) + (threadIdx.y*8192)) + (threadIdx.x*256)) + (axis2.axis3.fused.outer*16)), 1, 16)] = A[ramp(((((blockIdx.x*262144) + (axis2.axis3.fused.outer*16384)) + (threadIdx.y*512)) + (threadIdx.x*16)), 1, 16)]
    }
    for (axis0: int32, 0, 1024) {
      for (axis1: int32, 0, 1024) {
        for (axis2: int32, 0, 16) {
          for (axis3: int32, 0, 16) {
            let cse_var_1: int32 = ((((axis0*262144) + (axis1*256)) + (axis2*16)) + axis3)
            A.global.shared_1: Buffer(A.global.shared, int8, [268435456], [], scope="shared")[cse_var_1] = A.global_1[cse_var_1]
          }
        }
      }
    }
    for (axis0_1: int32, 0, 1024) {
      for (axis1_1: int32, 0, 1024) {
        for (axis2_1: int32, 0, 16) {
          for (axis3_1: int32, 0, 16) {
            let cse_var_2: int32 = ((((axis0_1*262144) + (axis1_1*256)) + (axis2_1*16)) + axis3_1)
            A.global.shared.local_1: Buffer(A.global.shared.local, int8, [268435456], [], scope="local")[cse_var_2] = A.global.shared_1[cse_var_2]
          }
        }
      }
    }
    attr [IterVar(blockIdx.x_1: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1024;
    attr [IterVar(threadIdx.y_1: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 32;
    attr [IterVar(threadIdx.x_1: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    for (axis2.axis3.fused.outer_1: int32, 0, 16) {
      A.global_2: Buffer(A.global, int8, [268435456], [])[ramp(((((blockIdx.x_1*262144) + (threadIdx.y_1*8192)) + (threadIdx.x_1*256)) + (axis2.axis3.fused.outer_1*16)), 1, 16)] = B[ramp(((((blockIdx.x_1*262144) + (axis2.axis3.fused.outer_1*16384)) + (threadIdx.y_1*512)) + (threadIdx.x_1*16)), 1, 16)]
    }
    for (axis0_2: int32, 0, 1024) {
      for (axis1_2: int32, 0, 1024) {
        for (axis2_2: int32, 0, 16) {
          for (axis3_2: int32, 0, 16) {
            let cse_var_3: int32 = ((((axis0_2*262144) + (axis1_2*256)) + (axis2_2*16)) + axis3_2)
            A.global.shared_2: Buffer(A.global.shared, int8, [268435456], [], scope="shared")[cse_var_3] = A.global_2[cse_var_3]
          }
        }
      }
    }
    for (axis0_3: int32, 0, 1024) {
      for (axis1_3: int32, 0, 1024) {
        for (axis2_3: int32, 0, 16) {
          for (axis3_3: int32, 0, 16) {
            let cse_var_4: int32 = ((((axis0_3*262144) + (axis1_3*256)) + (axis2_3*16)) + axis3_3)
            B.global.shared.local_1: Buffer(B.global.shared.local, int8, [268435456], [], scope="local")[cse_var_4] = A.global.shared_2[cse_var_4]
          }
        }
      }
    }
    for (axis0_4: int32, 0, 1024) {
      for (axis1_4: int32, 0, 1024) {
        for (axis2_4: int32, 0, 16) {
          for (axis3_4: int32, 0, 16) {
            C.local_1: Buffer(C.local, int32, [268435456], [], scope="local")[((((axis0_4*262144) + (axis1_4*256)) + (axis2_4*16)) + axis3_4)] = 0
            for (rk.outer: int32, 0, 1024) {
              for (rk.inner: int32, 0, 16) {
                let cse_var_8: int32 = (axis0_4*262144)
                let cse_var_7: int32 = (axis2_4*16)
                let cse_var_6: int32 = (rk.outer*256)
                let cse_var_5: int32 = (((cse_var_8 + (axis1_4*256)) + cse_var_7) + axis3_4)
                C.local_1[cse_var_5] = (C.local_1[cse_var_5] + (cast(int32, A.global.shared.local_1[(((cse_var_8 + cse_var_6) + cse_var_7) + rk.inner)])*cast(int32, B.global.shared.local_1[((((axis1_4*262144) + cse_var_6) + (axis3_4*16)) + rk.inner)])))
              }
            }
          }
        }
      }
    }
    for (axis0_5: int32, 0, 1024) {
      for (axis1_5: int32, 0, 1024) {
        for (axis2_5: int32, 0, 16) {
          for (axis3_5: int32, 0, 16) {
            let cse_var_9: int32 = ((((axis0_5*262144) + (axis1_5*256)) + (axis2_5*16)) + axis3_5)
            C.local.global_1: Buffer(C.local.global, int32, [268435456], [])[cse_var_9] = C.local_1[cse_var_9]
          }
        }
      }
    }
    attr [IterVar(blockIdx.x_2: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1024;
    attr [IterVar(threadIdx.y_2: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 32;
    attr [IterVar(threadIdx.x_2: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    for (ii.jj.fused.outer.inner: int32, 0, 16) {
      C[ramp(((((blockIdx.x_2*262144) + (threadIdx.y_2*8192)) + (threadIdx.x_2*256)) + (ii.jj.fused.outer.inner*16)), 1, 16)] = C.local.global_1[ramp((((blockIdx.x_2*262144) + (floormod((((threadIdx.y_2*512) + (threadIdx.x_2*16)) + ii.jj.fused.outer.inner), 1024)*256)) + (floordiv(threadIdx.y_2, 2)*16)), 1, 16)]
    }
  }
}

