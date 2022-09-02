@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [268435456], []),
             B: Buffer(B_2: Pointer(float32), float32, [268435456], []),
             C: Buffer(C_2: Pointer(float32), float32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [16384, 16384], []), B_1: B_3: Buffer(B_2, float32, [16384, 16384], []), C_1: C_3: Buffer(C_2, float32, [16384, 16384], [])} {
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 128;
  allocate(C.local: Pointer(local float32), float32, [64]), storage_scope = local;
  allocate(A.shared: Pointer(shared float32), float32, [2048]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float32), float32, [2048]), storage_scope = shared;
  allocate(A.shared.local: Pointer(local float32), float32, [8]), storage_scope = local;
  allocate(B.shared.local: Pointer(local float32), float32, [8]), storage_scope = local;
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 128;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 16;
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 16 {
    for (ii.c.init: int32, 0, 4) {
      for (jj.c.init: int32, 0, 4) {
        let cse_var_1: int32 = ((ii.c.init*4) + jj.c.init)
         {
          C.local_1: Buffer(C.local, float32, [8192], [], scope="local", align=64)[cse_var_1] = 0f32
          C.local_1[(cse_var_1 + 32)] = 0f32
          C.local_1[(cse_var_1 + 16)] = 0f32
          C.local_1[(cse_var_1 + 48)] = 0f32
        }
      }
    }
    for (k.outer: int32, 0, 1024) {
      for (ax0: int32, 0, 16) {
        for (ax1: int32, 0, 128) {
          A.shared_1: Buffer(A.shared, float32, [2048], [], scope="shared")[((ax0*128) + ax1)] = A[((((k.outer*262144) + (ax0*16384)) + (blockIdx.x*128)) + ax1)]
        }
      }
      for (ax0_1: int32, 0, 16) {
        for (ax1_1: int32, 0, 128) {
          B.shared_1: Buffer(B.shared, float32, [2048], [], scope="shared")[((ax0_1*128) + ax1_1)] = B[((((k.outer*262144) + (ax0_1*16384)) + (blockIdx.y*128)) + ax1_1)]
        }
      }
      for (ax1_2: int32, 0, 4) {
        A.shared.local_1: Buffer(A.shared.local, float32, [16], [], scope="local", align=16)[ax1_2] = A.shared_1[((threadIdx.x*4) + ax1_2)]
        A.shared.local_1[(ax1_2 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_2) + 64)]
      }
      for (ax1_3: int32, 0, 4) {
        B.shared.local_1: Buffer(B.shared.local, float32, [16], [], scope="local", align=16)[ax1_3] = B.shared_1[((threadIdx.y*4) + ax1_3)]
        B.shared.local_1[(ax1_3 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_3) + 64)]
      }
      for (ii.c: int32, 0, 4) {
        for (jj.c: int32, 0, 4) {
          let cse_var_7: int32 = (jj.c + 4)
          let cse_var_6: int32 = (ii.c + 4)
          let cse_var_5: int32 = ((ii.c*4) + jj.c)
          let cse_var_4: int32 = (cse_var_5 + 48)
          let cse_var_3: int32 = (cse_var_5 + 32)
          let cse_var_2: int32 = (cse_var_5 + 16)
           {
            C.local_1[cse_var_5] = (C.local_1[cse_var_5] + (A.shared.local_1[jj.c]*B.shared.local_1[ii.c]))
            C.local_1[cse_var_3] = (C.local_1[cse_var_3] + (A.shared.local_1[jj.c]*B.shared.local_1[cse_var_6]))
            C.local_1[cse_var_2] = (C.local_1[cse_var_2] + (A.shared.local_1[cse_var_7]*B.shared.local_1[ii.c]))
            C.local_1[cse_var_4] = (C.local_1[cse_var_4] + (A.shared.local_1[cse_var_7]*B.shared.local_1[cse_var_6]))
          }
        }
      }
      for (ax1_4: int32, 0, 4) {
        A.shared.local_1[ax1_4] = A.shared_1[(((threadIdx.x*4) + ax1_4) + 128)]
        A.shared.local_1[(ax1_4 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_4) + 192)]
      }
      for (ax1_5: int32, 0, 4) {
        B.shared.local_1[ax1_5] = B.shared_1[(((threadIdx.y*4) + ax1_5) + 128)]
        B.shared.local_1[(ax1_5 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_5) + 192)]
      }
      for (ii.c_1: int32, 0, 4) {
        for (jj.c_1: int32, 0, 4) {
          let cse_var_13: int32 = (jj.c_1 + 4)
          let cse_var_12: int32 = (ii.c_1 + 4)
          let cse_var_11: int32 = ((ii.c_1*4) + jj.c_1)
          let cse_var_10: int32 = (cse_var_11 + 48)
          let cse_var_9: int32 = (cse_var_11 + 32)
          let cse_var_8: int32 = (cse_var_11 + 16)
           {
            C.local_1[cse_var_11] = (C.local_1[cse_var_11] + (A.shared.local_1[jj.c_1]*B.shared.local_1[ii.c_1]))
            C.local_1[cse_var_9] = (C.local_1[cse_var_9] + (A.shared.local_1[jj.c_1]*B.shared.local_1[cse_var_12]))
            C.local_1[cse_var_8] = (C.local_1[cse_var_8] + (A.shared.local_1[cse_var_13]*B.shared.local_1[ii.c_1]))
            C.local_1[cse_var_10] = (C.local_1[cse_var_10] + (A.shared.local_1[cse_var_13]*B.shared.local_1[cse_var_12]))
          }
        }
      }
      for (ax1_6: int32, 0, 4) {
        A.shared.local_1[ax1_6] = A.shared_1[(((threadIdx.x*4) + ax1_6) + 256)]
        A.shared.local_1[(ax1_6 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_6) + 320)]
      }
      for (ax1_7: int32, 0, 4) {
        B.shared.local_1[ax1_7] = B.shared_1[(((threadIdx.y*4) + ax1_7) + 256)]
        B.shared.local_1[(ax1_7 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_7) + 320)]
      }
      for (ii.c_2: int32, 0, 4) {
        for (jj.c_2: int32, 0, 4) {
          let cse_var_19: int32 = (jj.c_2 + 4)
          let cse_var_18: int32 = (ii.c_2 + 4)
          let cse_var_17: int32 = ((ii.c_2*4) + jj.c_2)
          let cse_var_16: int32 = (cse_var_17 + 48)
          let cse_var_15: int32 = (cse_var_17 + 32)
          let cse_var_14: int32 = (cse_var_17 + 16)
           {
            C.local_1[cse_var_17] = (C.local_1[cse_var_17] + (A.shared.local_1[jj.c_2]*B.shared.local_1[ii.c_2]))
            C.local_1[cse_var_15] = (C.local_1[cse_var_15] + (A.shared.local_1[jj.c_2]*B.shared.local_1[cse_var_18]))
            C.local_1[cse_var_14] = (C.local_1[cse_var_14] + (A.shared.local_1[cse_var_19]*B.shared.local_1[ii.c_2]))
            C.local_1[cse_var_16] = (C.local_1[cse_var_16] + (A.shared.local_1[cse_var_19]*B.shared.local_1[cse_var_18]))
          }
        }
      }
      for (ax1_8: int32, 0, 4) {
        A.shared.local_1[ax1_8] = A.shared_1[(((threadIdx.x*4) + ax1_8) + 384)]
        A.shared.local_1[(ax1_8 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_8) + 448)]
      }
      for (ax1_9: int32, 0, 4) {
        B.shared.local_1[ax1_9] = B.shared_1[(((threadIdx.y*4) + ax1_9) + 384)]
        B.shared.local_1[(ax1_9 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_9) + 448)]
      }
      for (ii.c_3: int32, 0, 4) {
        for (jj.c_3: int32, 0, 4) {
          let cse_var_25: int32 = (jj.c_3 + 4)
          let cse_var_24: int32 = (ii.c_3 + 4)
          let cse_var_23: int32 = ((ii.c_3*4) + jj.c_3)
          let cse_var_22: int32 = (cse_var_23 + 48)
          let cse_var_21: int32 = (cse_var_23 + 32)
          let cse_var_20: int32 = (cse_var_23 + 16)
           {
            C.local_1[cse_var_23] = (C.local_1[cse_var_23] + (A.shared.local_1[jj.c_3]*B.shared.local_1[ii.c_3]))
            C.local_1[cse_var_21] = (C.local_1[cse_var_21] + (A.shared.local_1[jj.c_3]*B.shared.local_1[cse_var_24]))
            C.local_1[cse_var_20] = (C.local_1[cse_var_20] + (A.shared.local_1[cse_var_25]*B.shared.local_1[ii.c_3]))
            C.local_1[cse_var_22] = (C.local_1[cse_var_22] + (A.shared.local_1[cse_var_25]*B.shared.local_1[cse_var_24]))
          }
        }
      }
      for (ax1_10: int32, 0, 4) {
        A.shared.local_1[ax1_10] = A.shared_1[(((threadIdx.x*4) + ax1_10) + 512)]
        A.shared.local_1[(ax1_10 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_10) + 576)]
      }
      for (ax1_11: int32, 0, 4) {
        B.shared.local_1[ax1_11] = B.shared_1[(((threadIdx.y*4) + ax1_11) + 512)]
        B.shared.local_1[(ax1_11 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_11) + 576)]
      }
      for (ii.c_4: int32, 0, 4) {
        for (jj.c_4: int32, 0, 4) {
          let cse_var_31: int32 = (jj.c_4 + 4)
          let cse_var_30: int32 = (ii.c_4 + 4)
          let cse_var_29: int32 = ((ii.c_4*4) + jj.c_4)
          let cse_var_28: int32 = (cse_var_29 + 48)
          let cse_var_27: int32 = (cse_var_29 + 32)
          let cse_var_26: int32 = (cse_var_29 + 16)
           {
            C.local_1[cse_var_29] = (C.local_1[cse_var_29] + (A.shared.local_1[jj.c_4]*B.shared.local_1[ii.c_4]))
            C.local_1[cse_var_27] = (C.local_1[cse_var_27] + (A.shared.local_1[jj.c_4]*B.shared.local_1[cse_var_30]))
            C.local_1[cse_var_26] = (C.local_1[cse_var_26] + (A.shared.local_1[cse_var_31]*B.shared.local_1[ii.c_4]))
            C.local_1[cse_var_28] = (C.local_1[cse_var_28] + (A.shared.local_1[cse_var_31]*B.shared.local_1[cse_var_30]))
          }
        }
      }
      for (ax1_12: int32, 0, 4) {
        A.shared.local_1[ax1_12] = A.shared_1[(((threadIdx.x*4) + ax1_12) + 640)]
        A.shared.local_1[(ax1_12 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_12) + 704)]
      }
      for (ax1_13: int32, 0, 4) {
        B.shared.local_1[ax1_13] = B.shared_1[(((threadIdx.y*4) + ax1_13) + 640)]
        B.shared.local_1[(ax1_13 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_13) + 704)]
      }
      for (ii.c_5: int32, 0, 4) {
        for (jj.c_5: int32, 0, 4) {
          let cse_var_37: int32 = (jj.c_5 + 4)
          let cse_var_36: int32 = (ii.c_5 + 4)
          let cse_var_35: int32 = ((ii.c_5*4) + jj.c_5)
          let cse_var_34: int32 = (cse_var_35 + 48)
          let cse_var_33: int32 = (cse_var_35 + 32)
          let cse_var_32: int32 = (cse_var_35 + 16)
           {
            C.local_1[cse_var_35] = (C.local_1[cse_var_35] + (A.shared.local_1[jj.c_5]*B.shared.local_1[ii.c_5]))
            C.local_1[cse_var_33] = (C.local_1[cse_var_33] + (A.shared.local_1[jj.c_5]*B.shared.local_1[cse_var_36]))
            C.local_1[cse_var_32] = (C.local_1[cse_var_32] + (A.shared.local_1[cse_var_37]*B.shared.local_1[ii.c_5]))
            C.local_1[cse_var_34] = (C.local_1[cse_var_34] + (A.shared.local_1[cse_var_37]*B.shared.local_1[cse_var_36]))
          }
        }
      }
      for (ax1_14: int32, 0, 4) {
        A.shared.local_1[ax1_14] = A.shared_1[(((threadIdx.x*4) + ax1_14) + 768)]
        A.shared.local_1[(ax1_14 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_14) + 832)]
      }
      for (ax1_15: int32, 0, 4) {
        B.shared.local_1[ax1_15] = B.shared_1[(((threadIdx.y*4) + ax1_15) + 768)]
        B.shared.local_1[(ax1_15 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_15) + 832)]
      }
      for (ii.c_6: int32, 0, 4) {
        for (jj.c_6: int32, 0, 4) {
          let cse_var_43: int32 = (jj.c_6 + 4)
          let cse_var_42: int32 = (ii.c_6 + 4)
          let cse_var_41: int32 = ((ii.c_6*4) + jj.c_6)
          let cse_var_40: int32 = (cse_var_41 + 48)
          let cse_var_39: int32 = (cse_var_41 + 32)
          let cse_var_38: int32 = (cse_var_41 + 16)
           {
            C.local_1[cse_var_41] = (C.local_1[cse_var_41] + (A.shared.local_1[jj.c_6]*B.shared.local_1[ii.c_6]))
            C.local_1[cse_var_39] = (C.local_1[cse_var_39] + (A.shared.local_1[jj.c_6]*B.shared.local_1[cse_var_42]))
            C.local_1[cse_var_38] = (C.local_1[cse_var_38] + (A.shared.local_1[cse_var_43]*B.shared.local_1[ii.c_6]))
            C.local_1[cse_var_40] = (C.local_1[cse_var_40] + (A.shared.local_1[cse_var_43]*B.shared.local_1[cse_var_42]))
          }
        }
      }
      for (ax1_16: int32, 0, 4) {
        A.shared.local_1[ax1_16] = A.shared_1[(((threadIdx.x*4) + ax1_16) + 896)]
        A.shared.local_1[(ax1_16 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_16) + 960)]
      }
      for (ax1_17: int32, 0, 4) {
        B.shared.local_1[ax1_17] = B.shared_1[(((threadIdx.y*4) + ax1_17) + 896)]
        B.shared.local_1[(ax1_17 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_17) + 960)]
      }
      for (ii.c_7: int32, 0, 4) {
        for (jj.c_7: int32, 0, 4) {
          let cse_var_49: int32 = (jj.c_7 + 4)
          let cse_var_48: int32 = (ii.c_7 + 4)
          let cse_var_47: int32 = ((ii.c_7*4) + jj.c_7)
          let cse_var_46: int32 = (cse_var_47 + 48)
          let cse_var_45: int32 = (cse_var_47 + 32)
          let cse_var_44: int32 = (cse_var_47 + 16)
           {
            C.local_1[cse_var_47] = (C.local_1[cse_var_47] + (A.shared.local_1[jj.c_7]*B.shared.local_1[ii.c_7]))
            C.local_1[cse_var_45] = (C.local_1[cse_var_45] + (A.shared.local_1[jj.c_7]*B.shared.local_1[cse_var_48]))
            C.local_1[cse_var_44] = (C.local_1[cse_var_44] + (A.shared.local_1[cse_var_49]*B.shared.local_1[ii.c_7]))
            C.local_1[cse_var_46] = (C.local_1[cse_var_46] + (A.shared.local_1[cse_var_49]*B.shared.local_1[cse_var_48]))
          }
        }
      }
      for (ax1_18: int32, 0, 4) {
        A.shared.local_1[ax1_18] = A.shared_1[(((threadIdx.x*4) + ax1_18) + 1024)]
        A.shared.local_1[(ax1_18 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_18) + 1088)]
      }
      for (ax1_19: int32, 0, 4) {
        B.shared.local_1[ax1_19] = B.shared_1[(((threadIdx.y*4) + ax1_19) + 1024)]
        B.shared.local_1[(ax1_19 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_19) + 1088)]
      }
      for (ii.c_8: int32, 0, 4) {
        for (jj.c_8: int32, 0, 4) {
          let cse_var_55: int32 = (jj.c_8 + 4)
          let cse_var_54: int32 = (ii.c_8 + 4)
          let cse_var_53: int32 = ((ii.c_8*4) + jj.c_8)
          let cse_var_52: int32 = (cse_var_53 + 48)
          let cse_var_51: int32 = (cse_var_53 + 32)
          let cse_var_50: int32 = (cse_var_53 + 16)
           {
            C.local_1[cse_var_53] = (C.local_1[cse_var_53] + (A.shared.local_1[jj.c_8]*B.shared.local_1[ii.c_8]))
            C.local_1[cse_var_51] = (C.local_1[cse_var_51] + (A.shared.local_1[jj.c_8]*B.shared.local_1[cse_var_54]))
            C.local_1[cse_var_50] = (C.local_1[cse_var_50] + (A.shared.local_1[cse_var_55]*B.shared.local_1[ii.c_8]))
            C.local_1[cse_var_52] = (C.local_1[cse_var_52] + (A.shared.local_1[cse_var_55]*B.shared.local_1[cse_var_54]))
          }
        }
      }
      for (ax1_20: int32, 0, 4) {
        A.shared.local_1[ax1_20] = A.shared_1[(((threadIdx.x*4) + ax1_20) + 1152)]
        A.shared.local_1[(ax1_20 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_20) + 1216)]
      }
      for (ax1_21: int32, 0, 4) {
        B.shared.local_1[ax1_21] = B.shared_1[(((threadIdx.y*4) + ax1_21) + 1152)]
        B.shared.local_1[(ax1_21 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_21) + 1216)]
      }
      for (ii.c_9: int32, 0, 4) {
        for (jj.c_9: int32, 0, 4) {
          let cse_var_61: int32 = (jj.c_9 + 4)
          let cse_var_60: int32 = (ii.c_9 + 4)
          let cse_var_59: int32 = ((ii.c_9*4) + jj.c_9)
          let cse_var_58: int32 = (cse_var_59 + 48)
          let cse_var_57: int32 = (cse_var_59 + 32)
          let cse_var_56: int32 = (cse_var_59 + 16)
           {
            C.local_1[cse_var_59] = (C.local_1[cse_var_59] + (A.shared.local_1[jj.c_9]*B.shared.local_1[ii.c_9]))
            C.local_1[cse_var_57] = (C.local_1[cse_var_57] + (A.shared.local_1[jj.c_9]*B.shared.local_1[cse_var_60]))
            C.local_1[cse_var_56] = (C.local_1[cse_var_56] + (A.shared.local_1[cse_var_61]*B.shared.local_1[ii.c_9]))
            C.local_1[cse_var_58] = (C.local_1[cse_var_58] + (A.shared.local_1[cse_var_61]*B.shared.local_1[cse_var_60]))
          }
        }
      }
      for (ax1_22: int32, 0, 4) {
        A.shared.local_1[ax1_22] = A.shared_1[(((threadIdx.x*4) + ax1_22) + 1280)]
        A.shared.local_1[(ax1_22 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_22) + 1344)]
      }
      for (ax1_23: int32, 0, 4) {
        B.shared.local_1[ax1_23] = B.shared_1[(((threadIdx.y*4) + ax1_23) + 1280)]
        B.shared.local_1[(ax1_23 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_23) + 1344)]
      }
      for (ii.c_10: int32, 0, 4) {
        for (jj.c_10: int32, 0, 4) {
          let cse_var_67: int32 = (jj.c_10 + 4)
          let cse_var_66: int32 = (ii.c_10 + 4)
          let cse_var_65: int32 = ((ii.c_10*4) + jj.c_10)
          let cse_var_64: int32 = (cse_var_65 + 48)
          let cse_var_63: int32 = (cse_var_65 + 32)
          let cse_var_62: int32 = (cse_var_65 + 16)
           {
            C.local_1[cse_var_65] = (C.local_1[cse_var_65] + (A.shared.local_1[jj.c_10]*B.shared.local_1[ii.c_10]))
            C.local_1[cse_var_63] = (C.local_1[cse_var_63] + (A.shared.local_1[jj.c_10]*B.shared.local_1[cse_var_66]))
            C.local_1[cse_var_62] = (C.local_1[cse_var_62] + (A.shared.local_1[cse_var_67]*B.shared.local_1[ii.c_10]))
            C.local_1[cse_var_64] = (C.local_1[cse_var_64] + (A.shared.local_1[cse_var_67]*B.shared.local_1[cse_var_66]))
          }
        }
      }
      for (ax1_24: int32, 0, 4) {
        A.shared.local_1[ax1_24] = A.shared_1[(((threadIdx.x*4) + ax1_24) + 1408)]
        A.shared.local_1[(ax1_24 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_24) + 1472)]
      }
      for (ax1_25: int32, 0, 4) {
        B.shared.local_1[ax1_25] = B.shared_1[(((threadIdx.y*4) + ax1_25) + 1408)]
        B.shared.local_1[(ax1_25 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_25) + 1472)]
      }
      for (ii.c_11: int32, 0, 4) {
        for (jj.c_11: int32, 0, 4) {
          let cse_var_73: int32 = (jj.c_11 + 4)
          let cse_var_72: int32 = (ii.c_11 + 4)
          let cse_var_71: int32 = ((ii.c_11*4) + jj.c_11)
          let cse_var_70: int32 = (cse_var_71 + 48)
          let cse_var_69: int32 = (cse_var_71 + 32)
          let cse_var_68: int32 = (cse_var_71 + 16)
           {
            C.local_1[cse_var_71] = (C.local_1[cse_var_71] + (A.shared.local_1[jj.c_11]*B.shared.local_1[ii.c_11]))
            C.local_1[cse_var_69] = (C.local_1[cse_var_69] + (A.shared.local_1[jj.c_11]*B.shared.local_1[cse_var_72]))
            C.local_1[cse_var_68] = (C.local_1[cse_var_68] + (A.shared.local_1[cse_var_73]*B.shared.local_1[ii.c_11]))
            C.local_1[cse_var_70] = (C.local_1[cse_var_70] + (A.shared.local_1[cse_var_73]*B.shared.local_1[cse_var_72]))
          }
        }
      }
      for (ax1_26: int32, 0, 4) {
        A.shared.local_1[ax1_26] = A.shared_1[(((threadIdx.x*4) + ax1_26) + 1536)]
        A.shared.local_1[(ax1_26 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_26) + 1600)]
      }
      for (ax1_27: int32, 0, 4) {
        B.shared.local_1[ax1_27] = B.shared_1[(((threadIdx.y*4) + ax1_27) + 1536)]
        B.shared.local_1[(ax1_27 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_27) + 1600)]
      }
      for (ii.c_12: int32, 0, 4) {
        for (jj.c_12: int32, 0, 4) {
          let cse_var_79: int32 = (jj.c_12 + 4)
          let cse_var_78: int32 = (ii.c_12 + 4)
          let cse_var_77: int32 = ((ii.c_12*4) + jj.c_12)
          let cse_var_76: int32 = (cse_var_77 + 48)
          let cse_var_75: int32 = (cse_var_77 + 32)
          let cse_var_74: int32 = (cse_var_77 + 16)
           {
            C.local_1[cse_var_77] = (C.local_1[cse_var_77] + (A.shared.local_1[jj.c_12]*B.shared.local_1[ii.c_12]))
            C.local_1[cse_var_75] = (C.local_1[cse_var_75] + (A.shared.local_1[jj.c_12]*B.shared.local_1[cse_var_78]))
            C.local_1[cse_var_74] = (C.local_1[cse_var_74] + (A.shared.local_1[cse_var_79]*B.shared.local_1[ii.c_12]))
            C.local_1[cse_var_76] = (C.local_1[cse_var_76] + (A.shared.local_1[cse_var_79]*B.shared.local_1[cse_var_78]))
          }
        }
      }
      for (ax1_28: int32, 0, 4) {
        A.shared.local_1[ax1_28] = A.shared_1[(((threadIdx.x*4) + ax1_28) + 1664)]
        A.shared.local_1[(ax1_28 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_28) + 1728)]
      }
      for (ax1_29: int32, 0, 4) {
        B.shared.local_1[ax1_29] = B.shared_1[(((threadIdx.y*4) + ax1_29) + 1664)]
        B.shared.local_1[(ax1_29 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_29) + 1728)]
      }
      for (ii.c_13: int32, 0, 4) {
        for (jj.c_13: int32, 0, 4) {
          let cse_var_85: int32 = (jj.c_13 + 4)
          let cse_var_84: int32 = (ii.c_13 + 4)
          let cse_var_83: int32 = ((ii.c_13*4) + jj.c_13)
          let cse_var_82: int32 = (cse_var_83 + 48)
          let cse_var_81: int32 = (cse_var_83 + 32)
          let cse_var_80: int32 = (cse_var_83 + 16)
           {
            C.local_1[cse_var_83] = (C.local_1[cse_var_83] + (A.shared.local_1[jj.c_13]*B.shared.local_1[ii.c_13]))
            C.local_1[cse_var_81] = (C.local_1[cse_var_81] + (A.shared.local_1[jj.c_13]*B.shared.local_1[cse_var_84]))
            C.local_1[cse_var_80] = (C.local_1[cse_var_80] + (A.shared.local_1[cse_var_85]*B.shared.local_1[ii.c_13]))
            C.local_1[cse_var_82] = (C.local_1[cse_var_82] + (A.shared.local_1[cse_var_85]*B.shared.local_1[cse_var_84]))
          }
        }
      }
      for (ax1_30: int32, 0, 4) {
        A.shared.local_1[ax1_30] = A.shared_1[(((threadIdx.x*4) + ax1_30) + 1792)]
        A.shared.local_1[(ax1_30 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_30) + 1856)]
      }
      for (ax1_31: int32, 0, 4) {
        B.shared.local_1[ax1_31] = B.shared_1[(((threadIdx.y*4) + ax1_31) + 1792)]
        B.shared.local_1[(ax1_31 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_31) + 1856)]
      }
      for (ii.c_14: int32, 0, 4) {
        for (jj.c_14: int32, 0, 4) {
          let cse_var_91: int32 = (jj.c_14 + 4)
          let cse_var_90: int32 = (ii.c_14 + 4)
          let cse_var_89: int32 = ((ii.c_14*4) + jj.c_14)
          let cse_var_88: int32 = (cse_var_89 + 48)
          let cse_var_87: int32 = (cse_var_89 + 32)
          let cse_var_86: int32 = (cse_var_89 + 16)
           {
            C.local_1[cse_var_89] = (C.local_1[cse_var_89] + (A.shared.local_1[jj.c_14]*B.shared.local_1[ii.c_14]))
            C.local_1[cse_var_87] = (C.local_1[cse_var_87] + (A.shared.local_1[jj.c_14]*B.shared.local_1[cse_var_90]))
            C.local_1[cse_var_86] = (C.local_1[cse_var_86] + (A.shared.local_1[cse_var_91]*B.shared.local_1[ii.c_14]))
            C.local_1[cse_var_88] = (C.local_1[cse_var_88] + (A.shared.local_1[cse_var_91]*B.shared.local_1[cse_var_90]))
          }
        }
      }
      for (ax1_32: int32, 0, 4) {
        A.shared.local_1[ax1_32] = A.shared_1[(((threadIdx.x*4) + ax1_32) + 1920)]
        A.shared.local_1[(ax1_32 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_32) + 1984)]
      }
      for (ax1_33: int32, 0, 4) {
        B.shared.local_1[ax1_33] = B.shared_1[(((threadIdx.y*4) + ax1_33) + 1920)]
        B.shared.local_1[(ax1_33 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_33) + 1984)]
      }
      for (ii.c_15: int32, 0, 4) {
        for (jj.c_15: int32, 0, 4) {
          let cse_var_97: int32 = (jj.c_15 + 4)
          let cse_var_96: int32 = (ii.c_15 + 4)
          let cse_var_95: int32 = ((ii.c_15*4) + jj.c_15)
          let cse_var_94: int32 = (cse_var_95 + 48)
          let cse_var_93: int32 = (cse_var_95 + 32)
          let cse_var_92: int32 = (cse_var_95 + 16)
           {
            C.local_1[cse_var_95] = (C.local_1[cse_var_95] + (A.shared.local_1[jj.c_15]*B.shared.local_1[ii.c_15]))
            C.local_1[cse_var_93] = (C.local_1[cse_var_93] + (A.shared.local_1[jj.c_15]*B.shared.local_1[cse_var_96]))
            C.local_1[cse_var_92] = (C.local_1[cse_var_92] + (A.shared.local_1[cse_var_97]*B.shared.local_1[ii.c_15]))
            C.local_1[cse_var_94] = (C.local_1[cse_var_94] + (A.shared.local_1[cse_var_97]*B.shared.local_1[cse_var_96]))
          }
        }
      }
    }
    for (ii.inner.inner.inner: int32, 0, 4) {
      for (jj.inner.inner.inner: int32, 0, 4) {
        let cse_var_98: int32 = ((ii.inner.inner.inner*4) + jj.inner.inner.inner)
         {
          C[((((((blockIdx.y*2097152) + (threadIdx.y*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.x*128)) + (threadIdx.x*4)) + jj.inner.inner.inner)] = C.local_1[cse_var_98]
          C[(((((((blockIdx.y*2097152) + (threadIdx.y*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.x*128)) + (threadIdx.x*4)) + jj.inner.inner.inner) + 1048576)] = C.local_1[(cse_var_98 + 32)]
          C[(((((((blockIdx.y*2097152) + (threadIdx.y*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.x*128)) + (threadIdx.x*4)) + jj.inner.inner.inner) + 64)] = C.local_1[(cse_var_98 + 16)]
          C[(((((((blockIdx.y*2097152) + (threadIdx.y*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.x*128)) + (threadIdx.x*4)) + jj.inner.inner.inner) + 1048640)] = C.local_1[(cse_var_98 + 48)]
        }
      }
    }
  }
}

