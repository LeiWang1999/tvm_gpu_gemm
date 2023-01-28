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
    for (axis0: int32, 0, 1024) {
      for (axis1: int32, 0, 1024) {
        for (axis2: int32, 0, 16) {
          for (axis3: int32, 0, 16) {
            let cse_var_1: int32 = (axis0*262144)
            A.global_1: Buffer(A.global, int8, [268435456], [])[(((cse_var_1 + (axis1*256)) + (axis2*16)) + axis3)] = A[(((cse_var_1 + (axis2*16384)) + (axis1*16)) + axis3)]
          }
        }
      }
    }
    for (axis0_1: int32, 0, 1024) {
      for (axis1_1: int32, 0, 1024) {
        for (axis2_1: int32, 0, 16) {
          for (axis3_1: int32, 0, 16) {
            let cse_var_2: int32 = ((((axis0_1*262144) + (axis1_1*256)) + (axis2_1*16)) + axis3_1)
            A.global.shared_1: Buffer(A.global.shared, int8, [268435456], [], scope="shared")[cse_var_2] = A.global_1[cse_var_2]
          }
        }
      }
    }
    for (ax0: int32, 0, 16384) {
      for (ax1: int32, 0, 16384) {
        A.global.shared.local_1: Buffer(A.global.shared.local, int8, [268435456], [], scope="local")[((ax0*16384) + ax1)] = A.global.shared_1[((((floordiv(ax0, 16)*262144) + (floordiv(ax1, 16)*256)) + (floormod(ax0, 16)*16)) + floormod(ax1, 16))]
      }
    }
    for (axis0_2: int32, 0, 1024) {
      for (axis1_2: int32, 0, 1024) {
        for (axis2_2: int32, 0, 16) {
          for (axis3_2: int32, 0, 16) {
            let cse_var_3: int32 = (axis0_2*262144)
            A.global_2: Buffer(A.global, int8, [268435456], [])[(((cse_var_3 + (axis1_2*256)) + (axis2_2*16)) + axis3_2)] = B[(((cse_var_3 + (axis2_2*16384)) + (axis1_2*16)) + axis3_2)]
          }
        }
      }
    }
    for (axis0_3: int32, 0, 1024) {
      for (axis1_3: int32, 0, 1024) {
        for (axis2_3: int32, 0, 16) {
          for (axis3_3: int32, 0, 16) {
            let cse_var_4: int32 = ((((axis0_3*262144) + (axis1_3*256)) + (axis2_3*16)) + axis3_3)
            A.global.shared_2: Buffer(A.global.shared, int8, [268435456], [], scope="shared")[cse_var_4] = A.global_2[cse_var_4]
          }
        }
      }
    }
    for (ax0_1: int32, 0, 16384) {
      for (ax1_1: int32, 0, 16384) {
        B.global.shared.local_1: Buffer(B.global.shared.local, int8, [268435456], [], scope="local")[((ax0_1*16384) + ax1_1)] = A.global.shared_2[((((floordiv(ax0_1, 16)*262144) + (floordiv(ax1_1, 16)*256)) + (floormod(ax0_1, 16)*16)) + floormod(ax1_1, 16))]
      }
    }
    for (ii.c: int32, 0, 16384) {
      for (jj.c: int32, 0, 16384) {
        C.local_1: Buffer(C.local, int32, [268435456], [], scope="local")[((ii.c*16384) + jj.c)] = 0
        for (rk: int32, 0, 16384) {
          let cse_var_6: int32 = (ii.c*16384)
          let cse_var_5: int32 = (cse_var_6 + jj.c)
          C.local_1[cse_var_5] = (C.local_1[cse_var_5] + (cast(int32, A.global.shared.local_1[(cse_var_6 + rk)])*cast(int32, B.global.shared.local_1[((jj.c*16384) + rk)])))
        }
      }
    }
    for (ax0_2: int32, 0, 16384) {
      for (ax1_2: int32, 0, 16384) {
        let cse_var_7: int32 = ((ax0_2*16384) + ax1_2)
        C.local.global_1: Buffer(C.local.global, int32, [268435456], [])[cse_var_7] = C.local_1[cse_var_7]
      }
    }
    for (ii: int32, 0, 16384) {
      for (jj: int32, 0, 16384) {
        let cse_var_8: int32 = ((ii*16384) + jj)
        C[cse_var_8] = C.local.global_1[cse_var_8]
      }
    }
  }
}

