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
      for (ax1.outer: int32, 0, 2) {
        for (ax1.inner.inner: int32, 0, 4) {
          let cse_var_2: int32 = (ax1.outer*64)
          A.shared_1: Buffer(A.shared, float32, [2048], [], scope="shared")[((((threadIdx.y*128) + cse_var_2) + (threadIdx.x*4)) + ax1.inner.inner)] = A[((((((k.outer*262144) + (threadIdx.y*16384)) + (blockIdx.x*128)) + cse_var_2) + (threadIdx.x*4)) + ax1.inner.inner)]
        }
      }
      for (ax0: int32, 0, 16) {
        for (ax1: int32, 0, 128) {
          B.shared_1: Buffer(B.shared, float32, [2048], [], scope="shared")[((ax0*128) + ax1)] = B[((((k.outer*262144) + (ax0*16384)) + (blockIdx.y*128)) + ax1)]
        }
      }
      for (ax1_1: int32, 0, 4) {
        A.shared.local_1: Buffer(A.shared.local, float32, [16], [], scope="local", align=16)[ax1_1] = A.shared_1[((threadIdx.x*4) + ax1_1)]
        A.shared.local_1[(ax1_1 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_1) + 64)]
      }
      for (ax1_2: int32, 0, 4) {
        B.shared.local_1: Buffer(B.shared.local, float32, [16], [], scope="local", align=16)[ax1_2] = B.shared_1[((threadIdx.y*4) + ax1_2)]
        B.shared.local_1[(ax1_2 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_2) + 64)]
      }
      for (ii.c: int32, 0, 4) {
        for (jj.c: int32, 0, 4) {
          let cse_var_8: int32 = (jj.c + 4)
          let cse_var_7: int32 = (ii.c + 4)
          let cse_var_6: int32 = ((ii.c*4) + jj.c)
          let cse_var_5: int32 = (cse_var_6 + 48)
          let cse_var_4: int32 = (cse_var_6 + 32)
          let cse_var_3: int32 = (cse_var_6 + 16)
           {
            C.local_1[cse_var_6] = (C.local_1[cse_var_6] + (A.shared.local_1[jj.c]*B.shared.local_1[ii.c]))
            C.local_1[cse_var_4] = (C.local_1[cse_var_4] + (A.shared.local_1[jj.c]*B.shared.local_1[cse_var_7]))
            C.local_1[cse_var_3] = (C.local_1[cse_var_3] + (A.shared.local_1[cse_var_8]*B.shared.local_1[ii.c]))
            C.local_1[cse_var_5] = (C.local_1[cse_var_5] + (A.shared.local_1[cse_var_8]*B.shared.local_1[cse_var_7]))
          }
        }
      }
      for (ax1_3: int32, 0, 4) {
        A.shared.local_1[ax1_3] = A.shared_1[(((threadIdx.x*4) + ax1_3) + 128)]
        A.shared.local_1[(ax1_3 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_3) + 192)]
      }
      for (ax1_4: int32, 0, 4) {
        B.shared.local_1[ax1_4] = B.shared_1[(((threadIdx.y*4) + ax1_4) + 128)]
        B.shared.local_1[(ax1_4 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_4) + 192)]
      }
      for (ii.c_1: int32, 0, 4) {
        for (jj.c_1: int32, 0, 4) {
          let cse_var_14: int32 = (jj.c_1 + 4)
          let cse_var_13: int32 = (ii.c_1 + 4)
          let cse_var_12: int32 = ((ii.c_1*4) + jj.c_1)
          let cse_var_11: int32 = (cse_var_12 + 48)
          let cse_var_10: int32 = (cse_var_12 + 32)
          let cse_var_9: int32 = (cse_var_12 + 16)
           {
            C.local_1[cse_var_12] = (C.local_1[cse_var_12] + (A.shared.local_1[jj.c_1]*B.shared.local_1[ii.c_1]))
            C.local_1[cse_var_10] = (C.local_1[cse_var_10] + (A.shared.local_1[jj.c_1]*B.shared.local_1[cse_var_13]))
            C.local_1[cse_var_9] = (C.local_1[cse_var_9] + (A.shared.local_1[cse_var_14]*B.shared.local_1[ii.c_1]))
            C.local_1[cse_var_11] = (C.local_1[cse_var_11] + (A.shared.local_1[cse_var_14]*B.shared.local_1[cse_var_13]))
          }
        }
      }
      for (ax1_5: int32, 0, 4) {
        A.shared.local_1[ax1_5] = A.shared_1[(((threadIdx.x*4) + ax1_5) + 256)]
        A.shared.local_1[(ax1_5 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_5) + 320)]
      }
      for (ax1_6: int32, 0, 4) {
        B.shared.local_1[ax1_6] = B.shared_1[(((threadIdx.y*4) + ax1_6) + 256)]
        B.shared.local_1[(ax1_6 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_6) + 320)]
      }
      for (ii.c_2: int32, 0, 4) {
        for (jj.c_2: int32, 0, 4) {
          let cse_var_20: int32 = (jj.c_2 + 4)
          let cse_var_19: int32 = (ii.c_2 + 4)
          let cse_var_18: int32 = ((ii.c_2*4) + jj.c_2)
          let cse_var_17: int32 = (cse_var_18 + 48)
          let cse_var_16: int32 = (cse_var_18 + 32)
          let cse_var_15: int32 = (cse_var_18 + 16)
           {
            C.local_1[cse_var_18] = (C.local_1[cse_var_18] + (A.shared.local_1[jj.c_2]*B.shared.local_1[ii.c_2]))
            C.local_1[cse_var_16] = (C.local_1[cse_var_16] + (A.shared.local_1[jj.c_2]*B.shared.local_1[cse_var_19]))
            C.local_1[cse_var_15] = (C.local_1[cse_var_15] + (A.shared.local_1[cse_var_20]*B.shared.local_1[ii.c_2]))
            C.local_1[cse_var_17] = (C.local_1[cse_var_17] + (A.shared.local_1[cse_var_20]*B.shared.local_1[cse_var_19]))
          }
        }
      }
      for (ax1_7: int32, 0, 4) {
        A.shared.local_1[ax1_7] = A.shared_1[(((threadIdx.x*4) + ax1_7) + 384)]
        A.shared.local_1[(ax1_7 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_7) + 448)]
      }
      for (ax1_8: int32, 0, 4) {
        B.shared.local_1[ax1_8] = B.shared_1[(((threadIdx.y*4) + ax1_8) + 384)]
        B.shared.local_1[(ax1_8 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_8) + 448)]
      }
      for (ii.c_3: int32, 0, 4) {
        for (jj.c_3: int32, 0, 4) {
          let cse_var_26: int32 = (jj.c_3 + 4)
          let cse_var_25: int32 = (ii.c_3 + 4)
          let cse_var_24: int32 = ((ii.c_3*4) + jj.c_3)
          let cse_var_23: int32 = (cse_var_24 + 48)
          let cse_var_22: int32 = (cse_var_24 + 32)
          let cse_var_21: int32 = (cse_var_24 + 16)
           {
            C.local_1[cse_var_24] = (C.local_1[cse_var_24] + (A.shared.local_1[jj.c_3]*B.shared.local_1[ii.c_3]))
            C.local_1[cse_var_22] = (C.local_1[cse_var_22] + (A.shared.local_1[jj.c_3]*B.shared.local_1[cse_var_25]))
            C.local_1[cse_var_21] = (C.local_1[cse_var_21] + (A.shared.local_1[cse_var_26]*B.shared.local_1[ii.c_3]))
            C.local_1[cse_var_23] = (C.local_1[cse_var_23] + (A.shared.local_1[cse_var_26]*B.shared.local_1[cse_var_25]))
          }
        }
      }
      for (ax1_9: int32, 0, 4) {
        A.shared.local_1[ax1_9] = A.shared_1[(((threadIdx.x*4) + ax1_9) + 512)]
        A.shared.local_1[(ax1_9 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_9) + 576)]
      }
      for (ax1_10: int32, 0, 4) {
        B.shared.local_1[ax1_10] = B.shared_1[(((threadIdx.y*4) + ax1_10) + 512)]
        B.shared.local_1[(ax1_10 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_10) + 576)]
      }
      for (ii.c_4: int32, 0, 4) {
        for (jj.c_4: int32, 0, 4) {
          let cse_var_32: int32 = (jj.c_4 + 4)
          let cse_var_31: int32 = (ii.c_4 + 4)
          let cse_var_30: int32 = ((ii.c_4*4) + jj.c_4)
          let cse_var_29: int32 = (cse_var_30 + 48)
          let cse_var_28: int32 = (cse_var_30 + 32)
          let cse_var_27: int32 = (cse_var_30 + 16)
           {
            C.local_1[cse_var_30] = (C.local_1[cse_var_30] + (A.shared.local_1[jj.c_4]*B.shared.local_1[ii.c_4]))
            C.local_1[cse_var_28] = (C.local_1[cse_var_28] + (A.shared.local_1[jj.c_4]*B.shared.local_1[cse_var_31]))
            C.local_1[cse_var_27] = (C.local_1[cse_var_27] + (A.shared.local_1[cse_var_32]*B.shared.local_1[ii.c_4]))
            C.local_1[cse_var_29] = (C.local_1[cse_var_29] + (A.shared.local_1[cse_var_32]*B.shared.local_1[cse_var_31]))
          }
        }
      }
      for (ax1_11: int32, 0, 4) {
        A.shared.local_1[ax1_11] = A.shared_1[(((threadIdx.x*4) + ax1_11) + 640)]
        A.shared.local_1[(ax1_11 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_11) + 704)]
      }
      for (ax1_12: int32, 0, 4) {
        B.shared.local_1[ax1_12] = B.shared_1[(((threadIdx.y*4) + ax1_12) + 640)]
        B.shared.local_1[(ax1_12 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_12) + 704)]
      }
      for (ii.c_5: int32, 0, 4) {
        for (jj.c_5: int32, 0, 4) {
          let cse_var_38: int32 = (jj.c_5 + 4)
          let cse_var_37: int32 = (ii.c_5 + 4)
          let cse_var_36: int32 = ((ii.c_5*4) + jj.c_5)
          let cse_var_35: int32 = (cse_var_36 + 48)
          let cse_var_34: int32 = (cse_var_36 + 32)
          let cse_var_33: int32 = (cse_var_36 + 16)
           {
            C.local_1[cse_var_36] = (C.local_1[cse_var_36] + (A.shared.local_1[jj.c_5]*B.shared.local_1[ii.c_5]))
            C.local_1[cse_var_34] = (C.local_1[cse_var_34] + (A.shared.local_1[jj.c_5]*B.shared.local_1[cse_var_37]))
            C.local_1[cse_var_33] = (C.local_1[cse_var_33] + (A.shared.local_1[cse_var_38]*B.shared.local_1[ii.c_5]))
            C.local_1[cse_var_35] = (C.local_1[cse_var_35] + (A.shared.local_1[cse_var_38]*B.shared.local_1[cse_var_37]))
          }
        }
      }
      for (ax1_13: int32, 0, 4) {
        A.shared.local_1[ax1_13] = A.shared_1[(((threadIdx.x*4) + ax1_13) + 768)]
        A.shared.local_1[(ax1_13 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_13) + 832)]
      }
      for (ax1_14: int32, 0, 4) {
        B.shared.local_1[ax1_14] = B.shared_1[(((threadIdx.y*4) + ax1_14) + 768)]
        B.shared.local_1[(ax1_14 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_14) + 832)]
      }
      for (ii.c_6: int32, 0, 4) {
        for (jj.c_6: int32, 0, 4) {
          let cse_var_44: int32 = (jj.c_6 + 4)
          let cse_var_43: int32 = (ii.c_6 + 4)
          let cse_var_42: int32 = ((ii.c_6*4) + jj.c_6)
          let cse_var_41: int32 = (cse_var_42 + 48)
          let cse_var_40: int32 = (cse_var_42 + 32)
          let cse_var_39: int32 = (cse_var_42 + 16)
           {
            C.local_1[cse_var_42] = (C.local_1[cse_var_42] + (A.shared.local_1[jj.c_6]*B.shared.local_1[ii.c_6]))
            C.local_1[cse_var_40] = (C.local_1[cse_var_40] + (A.shared.local_1[jj.c_6]*B.shared.local_1[cse_var_43]))
            C.local_1[cse_var_39] = (C.local_1[cse_var_39] + (A.shared.local_1[cse_var_44]*B.shared.local_1[ii.c_6]))
            C.local_1[cse_var_41] = (C.local_1[cse_var_41] + (A.shared.local_1[cse_var_44]*B.shared.local_1[cse_var_43]))
          }
        }
      }
      for (ax1_15: int32, 0, 4) {
        A.shared.local_1[ax1_15] = A.shared_1[(((threadIdx.x*4) + ax1_15) + 896)]
        A.shared.local_1[(ax1_15 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_15) + 960)]
      }
      for (ax1_16: int32, 0, 4) {
        B.shared.local_1[ax1_16] = B.shared_1[(((threadIdx.y*4) + ax1_16) + 896)]
        B.shared.local_1[(ax1_16 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_16) + 960)]
      }
      for (ii.c_7: int32, 0, 4) {
        for (jj.c_7: int32, 0, 4) {
          let cse_var_50: int32 = (jj.c_7 + 4)
          let cse_var_49: int32 = (ii.c_7 + 4)
          let cse_var_48: int32 = ((ii.c_7*4) + jj.c_7)
          let cse_var_47: int32 = (cse_var_48 + 48)
          let cse_var_46: int32 = (cse_var_48 + 32)
          let cse_var_45: int32 = (cse_var_48 + 16)
           {
            C.local_1[cse_var_48] = (C.local_1[cse_var_48] + (A.shared.local_1[jj.c_7]*B.shared.local_1[ii.c_7]))
            C.local_1[cse_var_46] = (C.local_1[cse_var_46] + (A.shared.local_1[jj.c_7]*B.shared.local_1[cse_var_49]))
            C.local_1[cse_var_45] = (C.local_1[cse_var_45] + (A.shared.local_1[cse_var_50]*B.shared.local_1[ii.c_7]))
            C.local_1[cse_var_47] = (C.local_1[cse_var_47] + (A.shared.local_1[cse_var_50]*B.shared.local_1[cse_var_49]))
          }
        }
      }
      for (ax1_17: int32, 0, 4) {
        A.shared.local_1[ax1_17] = A.shared_1[(((threadIdx.x*4) + ax1_17) + 1024)]
        A.shared.local_1[(ax1_17 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_17) + 1088)]
      }
      for (ax1_18: int32, 0, 4) {
        B.shared.local_1[ax1_18] = B.shared_1[(((threadIdx.y*4) + ax1_18) + 1024)]
        B.shared.local_1[(ax1_18 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_18) + 1088)]
      }
      for (ii.c_8: int32, 0, 4) {
        for (jj.c_8: int32, 0, 4) {
          let cse_var_56: int32 = (jj.c_8 + 4)
          let cse_var_55: int32 = (ii.c_8 + 4)
          let cse_var_54: int32 = ((ii.c_8*4) + jj.c_8)
          let cse_var_53: int32 = (cse_var_54 + 48)
          let cse_var_52: int32 = (cse_var_54 + 32)
          let cse_var_51: int32 = (cse_var_54 + 16)
           {
            C.local_1[cse_var_54] = (C.local_1[cse_var_54] + (A.shared.local_1[jj.c_8]*B.shared.local_1[ii.c_8]))
            C.local_1[cse_var_52] = (C.local_1[cse_var_52] + (A.shared.local_1[jj.c_8]*B.shared.local_1[cse_var_55]))
            C.local_1[cse_var_51] = (C.local_1[cse_var_51] + (A.shared.local_1[cse_var_56]*B.shared.local_1[ii.c_8]))
            C.local_1[cse_var_53] = (C.local_1[cse_var_53] + (A.shared.local_1[cse_var_56]*B.shared.local_1[cse_var_55]))
          }
        }
      }
      for (ax1_19: int32, 0, 4) {
        A.shared.local_1[ax1_19] = A.shared_1[(((threadIdx.x*4) + ax1_19) + 1152)]
        A.shared.local_1[(ax1_19 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_19) + 1216)]
      }
      for (ax1_20: int32, 0, 4) {
        B.shared.local_1[ax1_20] = B.shared_1[(((threadIdx.y*4) + ax1_20) + 1152)]
        B.shared.local_1[(ax1_20 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_20) + 1216)]
      }
      for (ii.c_9: int32, 0, 4) {
        for (jj.c_9: int32, 0, 4) {
          let cse_var_62: int32 = (jj.c_9 + 4)
          let cse_var_61: int32 = (ii.c_9 + 4)
          let cse_var_60: int32 = ((ii.c_9*4) + jj.c_9)
          let cse_var_59: int32 = (cse_var_60 + 48)
          let cse_var_58: int32 = (cse_var_60 + 32)
          let cse_var_57: int32 = (cse_var_60 + 16)
           {
            C.local_1[cse_var_60] = (C.local_1[cse_var_60] + (A.shared.local_1[jj.c_9]*B.shared.local_1[ii.c_9]))
            C.local_1[cse_var_58] = (C.local_1[cse_var_58] + (A.shared.local_1[jj.c_9]*B.shared.local_1[cse_var_61]))
            C.local_1[cse_var_57] = (C.local_1[cse_var_57] + (A.shared.local_1[cse_var_62]*B.shared.local_1[ii.c_9]))
            C.local_1[cse_var_59] = (C.local_1[cse_var_59] + (A.shared.local_1[cse_var_62]*B.shared.local_1[cse_var_61]))
          }
        }
      }
      for (ax1_21: int32, 0, 4) {
        A.shared.local_1[ax1_21] = A.shared_1[(((threadIdx.x*4) + ax1_21) + 1280)]
        A.shared.local_1[(ax1_21 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_21) + 1344)]
      }
      for (ax1_22: int32, 0, 4) {
        B.shared.local_1[ax1_22] = B.shared_1[(((threadIdx.y*4) + ax1_22) + 1280)]
        B.shared.local_1[(ax1_22 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_22) + 1344)]
      }
      for (ii.c_10: int32, 0, 4) {
        for (jj.c_10: int32, 0, 4) {
          let cse_var_68: int32 = (jj.c_10 + 4)
          let cse_var_67: int32 = (ii.c_10 + 4)
          let cse_var_66: int32 = ((ii.c_10*4) + jj.c_10)
          let cse_var_65: int32 = (cse_var_66 + 48)
          let cse_var_64: int32 = (cse_var_66 + 32)
          let cse_var_63: int32 = (cse_var_66 + 16)
           {
            C.local_1[cse_var_66] = (C.local_1[cse_var_66] + (A.shared.local_1[jj.c_10]*B.shared.local_1[ii.c_10]))
            C.local_1[cse_var_64] = (C.local_1[cse_var_64] + (A.shared.local_1[jj.c_10]*B.shared.local_1[cse_var_67]))
            C.local_1[cse_var_63] = (C.local_1[cse_var_63] + (A.shared.local_1[cse_var_68]*B.shared.local_1[ii.c_10]))
            C.local_1[cse_var_65] = (C.local_1[cse_var_65] + (A.shared.local_1[cse_var_68]*B.shared.local_1[cse_var_67]))
          }
        }
      }
      for (ax1_23: int32, 0, 4) {
        A.shared.local_1[ax1_23] = A.shared_1[(((threadIdx.x*4) + ax1_23) + 1408)]
        A.shared.local_1[(ax1_23 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_23) + 1472)]
      }
      for (ax1_24: int32, 0, 4) {
        B.shared.local_1[ax1_24] = B.shared_1[(((threadIdx.y*4) + ax1_24) + 1408)]
        B.shared.local_1[(ax1_24 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_24) + 1472)]
      }
      for (ii.c_11: int32, 0, 4) {
        for (jj.c_11: int32, 0, 4) {
          let cse_var_74: int32 = (jj.c_11 + 4)
          let cse_var_73: int32 = (ii.c_11 + 4)
          let cse_var_72: int32 = ((ii.c_11*4) + jj.c_11)
          let cse_var_71: int32 = (cse_var_72 + 48)
          let cse_var_70: int32 = (cse_var_72 + 32)
          let cse_var_69: int32 = (cse_var_72 + 16)
           {
            C.local_1[cse_var_72] = (C.local_1[cse_var_72] + (A.shared.local_1[jj.c_11]*B.shared.local_1[ii.c_11]))
            C.local_1[cse_var_70] = (C.local_1[cse_var_70] + (A.shared.local_1[jj.c_11]*B.shared.local_1[cse_var_73]))
            C.local_1[cse_var_69] = (C.local_1[cse_var_69] + (A.shared.local_1[cse_var_74]*B.shared.local_1[ii.c_11]))
            C.local_1[cse_var_71] = (C.local_1[cse_var_71] + (A.shared.local_1[cse_var_74]*B.shared.local_1[cse_var_73]))
          }
        }
      }
      for (ax1_25: int32, 0, 4) {
        A.shared.local_1[ax1_25] = A.shared_1[(((threadIdx.x*4) + ax1_25) + 1536)]
        A.shared.local_1[(ax1_25 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_25) + 1600)]
      }
      for (ax1_26: int32, 0, 4) {
        B.shared.local_1[ax1_26] = B.shared_1[(((threadIdx.y*4) + ax1_26) + 1536)]
        B.shared.local_1[(ax1_26 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_26) + 1600)]
      }
      for (ii.c_12: int32, 0, 4) {
        for (jj.c_12: int32, 0, 4) {
          let cse_var_80: int32 = (jj.c_12 + 4)
          let cse_var_79: int32 = (ii.c_12 + 4)
          let cse_var_78: int32 = ((ii.c_12*4) + jj.c_12)
          let cse_var_77: int32 = (cse_var_78 + 48)
          let cse_var_76: int32 = (cse_var_78 + 32)
          let cse_var_75: int32 = (cse_var_78 + 16)
           {
            C.local_1[cse_var_78] = (C.local_1[cse_var_78] + (A.shared.local_1[jj.c_12]*B.shared.local_1[ii.c_12]))
            C.local_1[cse_var_76] = (C.local_1[cse_var_76] + (A.shared.local_1[jj.c_12]*B.shared.local_1[cse_var_79]))
            C.local_1[cse_var_75] = (C.local_1[cse_var_75] + (A.shared.local_1[cse_var_80]*B.shared.local_1[ii.c_12]))
            C.local_1[cse_var_77] = (C.local_1[cse_var_77] + (A.shared.local_1[cse_var_80]*B.shared.local_1[cse_var_79]))
          }
        }
      }
      for (ax1_27: int32, 0, 4) {
        A.shared.local_1[ax1_27] = A.shared_1[(((threadIdx.x*4) + ax1_27) + 1664)]
        A.shared.local_1[(ax1_27 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_27) + 1728)]
      }
      for (ax1_28: int32, 0, 4) {
        B.shared.local_1[ax1_28] = B.shared_1[(((threadIdx.y*4) + ax1_28) + 1664)]
        B.shared.local_1[(ax1_28 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_28) + 1728)]
      }
      for (ii.c_13: int32, 0, 4) {
        for (jj.c_13: int32, 0, 4) {
          let cse_var_86: int32 = (jj.c_13 + 4)
          let cse_var_85: int32 = (ii.c_13 + 4)
          let cse_var_84: int32 = ((ii.c_13*4) + jj.c_13)
          let cse_var_83: int32 = (cse_var_84 + 48)
          let cse_var_82: int32 = (cse_var_84 + 32)
          let cse_var_81: int32 = (cse_var_84 + 16)
           {
            C.local_1[cse_var_84] = (C.local_1[cse_var_84] + (A.shared.local_1[jj.c_13]*B.shared.local_1[ii.c_13]))
            C.local_1[cse_var_82] = (C.local_1[cse_var_82] + (A.shared.local_1[jj.c_13]*B.shared.local_1[cse_var_85]))
            C.local_1[cse_var_81] = (C.local_1[cse_var_81] + (A.shared.local_1[cse_var_86]*B.shared.local_1[ii.c_13]))
            C.local_1[cse_var_83] = (C.local_1[cse_var_83] + (A.shared.local_1[cse_var_86]*B.shared.local_1[cse_var_85]))
          }
        }
      }
      for (ax1_29: int32, 0, 4) {
        A.shared.local_1[ax1_29] = A.shared_1[(((threadIdx.x*4) + ax1_29) + 1792)]
        A.shared.local_1[(ax1_29 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_29) + 1856)]
      }
      for (ax1_30: int32, 0, 4) {
        B.shared.local_1[ax1_30] = B.shared_1[(((threadIdx.y*4) + ax1_30) + 1792)]
        B.shared.local_1[(ax1_30 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_30) + 1856)]
      }
      for (ii.c_14: int32, 0, 4) {
        for (jj.c_14: int32, 0, 4) {
          let cse_var_92: int32 = (jj.c_14 + 4)
          let cse_var_91: int32 = (ii.c_14 + 4)
          let cse_var_90: int32 = ((ii.c_14*4) + jj.c_14)
          let cse_var_89: int32 = (cse_var_90 + 48)
          let cse_var_88: int32 = (cse_var_90 + 32)
          let cse_var_87: int32 = (cse_var_90 + 16)
           {
            C.local_1[cse_var_90] = (C.local_1[cse_var_90] + (A.shared.local_1[jj.c_14]*B.shared.local_1[ii.c_14]))
            C.local_1[cse_var_88] = (C.local_1[cse_var_88] + (A.shared.local_1[jj.c_14]*B.shared.local_1[cse_var_91]))
            C.local_1[cse_var_87] = (C.local_1[cse_var_87] + (A.shared.local_1[cse_var_92]*B.shared.local_1[ii.c_14]))
            C.local_1[cse_var_89] = (C.local_1[cse_var_89] + (A.shared.local_1[cse_var_92]*B.shared.local_1[cse_var_91]))
          }
        }
      }
      for (ax1_31: int32, 0, 4) {
        A.shared.local_1[ax1_31] = A.shared_1[(((threadIdx.x*4) + ax1_31) + 1920)]
        A.shared.local_1[(ax1_31 + 4)] = A.shared_1[(((threadIdx.x*4) + ax1_31) + 1984)]
      }
      for (ax1_32: int32, 0, 4) {
        B.shared.local_1[ax1_32] = B.shared_1[(((threadIdx.y*4) + ax1_32) + 1920)]
        B.shared.local_1[(ax1_32 + 4)] = B.shared_1[(((threadIdx.y*4) + ax1_32) + 1984)]
      }
      for (ii.c_15: int32, 0, 4) {
        for (jj.c_15: int32, 0, 4) {
          let cse_var_98: int32 = (jj.c_15 + 4)
          let cse_var_97: int32 = (ii.c_15 + 4)
          let cse_var_96: int32 = ((ii.c_15*4) + jj.c_15)
          let cse_var_95: int32 = (cse_var_96 + 48)
          let cse_var_94: int32 = (cse_var_96 + 32)
          let cse_var_93: int32 = (cse_var_96 + 16)
           {
            C.local_1[cse_var_96] = (C.local_1[cse_var_96] + (A.shared.local_1[jj.c_15]*B.shared.local_1[ii.c_15]))
            C.local_1[cse_var_94] = (C.local_1[cse_var_94] + (A.shared.local_1[jj.c_15]*B.shared.local_1[cse_var_97]))
            C.local_1[cse_var_93] = (C.local_1[cse_var_93] + (A.shared.local_1[cse_var_98]*B.shared.local_1[ii.c_15]))
            C.local_1[cse_var_95] = (C.local_1[cse_var_95] + (A.shared.local_1[cse_var_98]*B.shared.local_1[cse_var_97]))
          }
        }
      }
    }
    for (ii.inner.inner.inner: int32, 0, 4) {
      for (jj.inner.inner.inner: int32, 0, 4) {
        let cse_var_99: int32 = ((ii.inner.inner.inner*4) + jj.inner.inner.inner)
         {
          C[((((((blockIdx.y*2097152) + (threadIdx.y*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.x*128)) + (threadIdx.x*4)) + jj.inner.inner.inner)] = C.local_1[cse_var_99]
          C[(((((((blockIdx.y*2097152) + (threadIdx.y*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.x*128)) + (threadIdx.x*4)) + jj.inner.inner.inner) + 1048576)] = C.local_1[(cse_var_99 + 32)]
          C[(((((((blockIdx.y*2097152) + (threadIdx.y*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.x*128)) + (threadIdx.x*4)) + jj.inner.inner.inner) + 64)] = C.local_1[(cse_var_99 + 16)]
          C[(((((((blockIdx.y*2097152) + (threadIdx.y*65536)) + (ii.inner.inner.inner*16384)) + (blockIdx.x*128)) + (threadIdx.x*4)) + jj.inner.inner.inner) + 1048640)] = C.local_1[(cse_var_99 + 48)]
        }
      }
    }
  }
}

