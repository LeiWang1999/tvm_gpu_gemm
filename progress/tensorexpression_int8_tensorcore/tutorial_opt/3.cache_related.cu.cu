#[version = "0.0.5"]
@main = primfn(A_1: handle, W_1: handle, Conv_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [16, 14, 14, 16, 16, 16], []),
             W: Buffer(W_2: Pointer(float16), float16, [3, 3, 16, 32, 16, 16], []),
             Conv: Buffer(Conv_2: Pointer(float32), float32, [16, 14, 14, 32, 16, 16], [])}
  buffer_map = {A_1: A, W_1: W, Conv_1: Conv} {
  allocate(Apad.shared: Pointer(shared float16), float16, [16777216]), storage_scope = shared;
  allocate(Apad.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [16777216]), storage_scope = wmma.matrix_a;
  allocate(W.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [1179648]), storage_scope = wmma.matrix_b;
  allocate(Conv.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [25690112]), storage_scope = wmma.accumulator {
    for (ax0: int32, 0, 16) {
      for (ax1: int32, 0, 16) {
        for (ax2: int32, 0, 16) {
          for (ax3: int32, 0, 16) {
            for (ax4: int32, 0, 16) {
              for (ax5: int32, 0, 16) {
                let cse_var_3: int32 = (ax2*4096)
                let cse_var_2: int32 = (ax3*256)
                let cse_var_1: int32 = (ax4*16)
                Apad.shared_1: Buffer(Apad.shared, float16, [16777216], [], scope="shared")[((((((ax0*1048576) + (ax1*65536)) + cse_var_3) + cse_var_2) + cse_var_1) + ax5)] = @tir.if_then_else(((((1 <= ax1) && (ax1 < 15)) && (1 <= ax2)) && (ax2 < 15)), A_3: Buffer(A_2, float16, [12845056], [])[(((((((ax0*802816) + (ax1*57344)) + cse_var_3) + cse_var_2) + cse_var_1) + ax5) - 61440)], 0f16, dtype=float16)
              }
            }
          }
        }
      }
    }
    for (ax0_1: int32, 0, 16) {
      for (ax1_1: int32, 0, 16) {
        for (ax2_1: int32, 0, 16) {
          for (ax3_1: int32, 0, 16) {
            for (ax4_1: int32, 0, 16) {
              for (ax5_1: int32, 0, 16) {
                let cse_var_4: int32 = ((((((ax0_1*1048576) + (ax1_1*65536)) + (ax2_1*4096)) + (ax3_1*256)) + (ax4_1*16)) + ax5_1)
                Apad.shared.wmma.matrix_a_1: Buffer(Apad.shared.wmma.matrix_a, float16, [16777216], [], scope="wmma.matrix_a")[cse_var_4] = Apad.shared_1[cse_var_4]
              }
            }
          }
        }
      }
    }
    for (ax0_2: int32, 0, 3) {
      for (ax1_2: int32, 0, 3) {
        for (ax2_2: int32, 0, 16) {
          for (ax3_2: int32, 0, 32) {
            for (ax4_2: int32, 0, 16) {
              for (ax5_2: int32, 0, 16) {
                let cse_var_5: int32 = ((((((ax0_2*393216) + (ax1_2*131072)) + (ax2_2*8192)) + (ax3_2*256)) + (ax4_2*16)) + ax5_2)
                Apad.shared_2: Buffer(Apad.shared, float16, [1179648], [], scope="shared")[cse_var_5] = W_3: Buffer(W_2, float16, [1179648], [])[cse_var_5]
              }
            }
          }
        }
      }
    }
    for (ax0_3: int32, 0, 3) {
      for (ax1_3: int32, 0, 3) {
        for (ax2_3: int32, 0, 16) {
          for (ax3_3: int32, 0, 32) {
            for (ax4_3: int32, 0, 16) {
              for (ax5_3: int32, 0, 16) {
                let cse_var_6: int32 = ((((((ax0_3*393216) + (ax1_3*131072)) + (ax2_3*8192)) + (ax3_3*256)) + (ax4_3*16)) + ax5_3)
                W.shared.wmma.matrix_b_1: Buffer(W.shared.wmma.matrix_b, float16, [1179648], [], scope="wmma.matrix_b")[cse_var_6] = Apad.shared_2[cse_var_6]
              }
            }
          }
        }
      }
    }
    for (n.c: int32, 0, 16) {
      for (h.c: int32, 0, 14) {
        for (w.c: int32, 0, 14) {
          for (o.c: int32, 0, 32) {
            for (nn.c: int32, 0, 16) {
              for (oo.c: int32, 0, 16) {
                Conv.wmma.accumulator_1: Buffer(Conv.wmma.accumulator, float32, [25690112], [], scope="wmma.accumulator")[((((((n.c*1605632) + (h.c*114688)) + (w.c*8192)) + (o.c*256)) + (nn.c*16)) + oo.c)] = 0f32
                for (ic: int32, 0, 16) {
                  for (kh: int32, 0, 3) {
                    for (kw: int32, 0, 3) {
                      for (ii: int32, 0, 16) {
                        let cse_var_9: int32 = (o.c*256)
                        let cse_var_8: int32 = (nn.c*16)
                        let cse_var_7: int32 = ((((((n.c*1605632) + (h.c*114688)) + (w.c*8192)) + cse_var_9) + cse_var_8) + oo.c)
                        Conv.wmma.accumulator_1[cse_var_7] = (Conv.wmma.accumulator_1[cse_var_7] + (cast(float32, Apad.shared.wmma.matrix_a_1[((((((((n.c*1048576) + (h.c*65536)) + (kh*65536)) + (w.c*4096)) + (kw*4096)) + (ic*256)) + cse_var_8) + ii)])*cast(float32, W.shared.wmma.matrix_b_1[((((((kh*393216) + (kw*131072)) + (ic*8192)) + cse_var_9) + (ii*16)) + oo.c)])))
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    for (n: int32, 0, 16) {
      for (h: int32, 0, 14) {
        for (w: int32, 0, 14) {
          for (o: int32, 0, 32) {
            for (nn: int32, 0, 16) {
              for (oo: int32, 0, 16) {
                let cse_var_10: int32 = ((((((n*1605632) + (h*114688)) + (w*8192)) + (o*256)) + (nn*16)) + oo)
                Conv_3: Buffer(Conv_2, float32, [25690112], [])[cse_var_10] = Conv.wmma.accumulator_1[cse_var_10]
              }
            }
          }
        }
      }
    }
  }
}

#[metadata]
{
  "root": 1, 
  "nodes": [
    {
      "type_key": ""
    }, 
    {
      "type_key": "Map", 
      "keys": [
        "IntImm"
      ], 
      "data": [2]
    }, 
    {
      "type_key": "Array", 
      "data": [3, 4]
    }, 
    {
      "type_key": "IntImm", 
      "attrs": {
        "dtype": "bool", 
        "span": "0", 
        "value": "1"
      }
    }, 
    {
      "type_key": "IntImm", 
      "attrs": {
        "dtype": "bool", 
        "span": "0", 
        "value": "1"
      }
    }
  ], 
  "b64ndarrays": [], 
  "attrs": {"tvm_version": "0.11.dev0"}
}