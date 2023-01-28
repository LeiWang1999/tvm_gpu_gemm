#[version = "0.0.5"]
@main = primfn(A_1: handle, W_1: handle, Conv_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [16, 14, 14, 16, 16, 16], []),
             W: Buffer(W_2: Pointer(float16), float16, [3, 3, 16, 32, 16, 16], []),
             Conv: Buffer(Conv_2: Pointer(float32), float32, [16, 14, 14, 32, 16, 16], [])}
  buffer_map = {A_1: A, W_1: W, Conv_1: Conv} {
  allocate(Apad.shared: Pointer(shared float16), float16, [294912]), storage_scope = shared;
  allocate(W.shared: Pointer(shared float16), float16, [294912]), storage_scope = shared {
    for (ax0: int32, 0, 8) {
      for (ax1: int32, 0, 3) {
        for (ax2: int32, 0, 3) {
          for (ax3: int32, 0, 16) {
            for (ax4: int32, 0, 16) {
              for (ax5: int32, 0, 16) {
                let cse_var_3: int32 = (ax2*4096)
                let cse_var_2: int32 = (ax3*256)
                let cse_var_1: int32 = (ax4*16)
                Apad.shared_1: Buffer(Apad.shared, float16, [294912], [], scope="shared")[((((((ax0*36864) + (ax1*12288)) + cse_var_3) + cse_var_2) + cse_var_1) + ax5)] = @tir.if_then_else(((((1 <= (floordiv(blockIdx.z: int32, 14) + ax1)) && ((floordiv(blockIdx.z, 14) + ax1) < 15)) && (1 <= (ax2 + floormod(blockIdx.z, 14)))) && ((ax2 + floormod(blockIdx.z, 14)) < 15)), A_3: Buffer(A_2, float16, [12845056], [])[(((((((((blockIdx.x: int32*6422528) + (ax0*802816)) + (ax1*57344)) + (blockIdx.z*4096)) + cse_var_3) + cse_var_2) + cse_var_1) + ax5) - 61440)], 0f16, dtype=float16)
              }
            }
          }
        }
      }
    }
    for (ax0_1: int32, 0, 3) {
      for (ax1_1: int32, 0, 3) {
        for (ax2_1: int32, 0, 16) {
          for (ax3_1: int32, 0, 8) {
            for (ax4_1: int32, 0, 16) {
              for (ax5_1: int32, 0, 16) {
                let cse_var_5: int32 = (ax3_1*256)
                let cse_var_4: int32 = (ax4_1*16)
                W.shared_1: Buffer(W.shared, float16, [294912], [], scope="shared")[((((((ax0_1*98304) + (ax1_1*32768)) + (ax2_1*2048)) + cse_var_5) + cse_var_4) + ax5_1)] = W_3: Buffer(W_2, float16, [1179648], [])[(((((((ax0_1*393216) + (ax1_1*131072)) + (ax2_1*8192)) + (blockIdx.y: int32*2048)) + cse_var_5) + cse_var_4) + ax5_1)]
              }
            }
          }
        }
      }
    }
    attr [IterVar(blockIdx.z, (nullptr), "ThreadIndex", "blockIdx.z")] "thread_extent" = 196;
    allocate(Conv.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [2048]), storage_scope = wmma.accumulator;
    allocate(Apad.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [512]), storage_scope = wmma.matrix_a;
    allocate(W.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [1024]), storage_scope = wmma.matrix_b;
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 2;
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 4;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 4;
    attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2 {
      for (n.c.init: int32, 0, 2) {
        for (o.c.init: int32, 0, 4) {
          for (nn.c.init: int32, 0, 16) {
            for (oo.c.init: int32, 0, 16) {
              Conv.wmma.accumulator_1: Buffer(Conv.wmma.accumulator, float32, [2048], [], scope="wmma.accumulator")[((((n.c.init*1024) + (o.c.init*256)) + (nn.c.init*16)) + oo.c.init)] = 0f32
            }
          }
        }
      }
      for (ic.outer: int32, 0, 8) {
        for (kh: int32, 0, 3) {
          for (ic.inner: int32, 0, 2) {
            for (kw: int32, 0, 3) {
              for (ax0_2: int32, 0, 2) {
                for (ax4_2: int32, 0, 16) {
                  for (ax5_2: int32, 0, 16) {
                    let cse_var_6: int32 = (ax4_2*16)
                    Apad.shared.wmma.matrix_a_1: Buffer(Apad.shared.wmma.matrix_a, float16, [512], [], scope="wmma.matrix_a")[(((ax0_2*256) + cse_var_6) + ax5_2)] = Apad.shared_1[((((((((threadIdx.y*73728) + (ax0_2*36864)) + (kh*12288)) + (kw*4096)) + (ic.outer*512)) + (ic.inner*256)) + cse_var_6) + ax5_2)]
                  }
                }
              }
              for (ax3_2: int32, 0, 4) {
                for (ax4_3: int32, 0, 16) {
                  for (ax5_3: int32, 0, 16) {
                    let cse_var_8: int32 = (ax3_2*256)
                    let cse_var_7: int32 = (ax4_3*16)
                    W.shared.wmma.matrix_b_1: Buffer(W.shared.wmma.matrix_b, float16, [1024], [], scope="wmma.matrix_b")[((cse_var_8 + cse_var_7) + ax5_3)] = W.shared_1[((((((((kh*98304) + (kw*32768)) + (ic.outer*4096)) + (ic.inner*2048)) + (threadIdx.z*1024)) + cse_var_8) + cse_var_7) + ax5_3)]
                  }
                }
              }
              for (n.c: int32, 0, 2) {
                for (o.c: int32, 0, 4) {
                  for (nn.c: int32, 0, 16) {
                    for (oo.c: int32, 0, 16) {
                      for (ii: int32, 0, 16) {
                        let cse_var_11: int32 = (o.c*256)
                        let cse_var_10: int32 = (nn.c*16)
                        let cse_var_9: int32 = ((((n.c*1024) + cse_var_11) + cse_var_10) + oo.c)
                        Conv.wmma.accumulator_1[cse_var_9] = (Conv.wmma.accumulator_1[cse_var_9] + (cast(float32, Apad.shared.wmma.matrix_a_1[(((n.c*256) + cse_var_10) + ii)])*cast(float32, W.shared.wmma.matrix_b_1[((cse_var_11 + (ii*16)) + oo.c)])))
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (n.inner: int32, 0, 2) {
        for (o.inner: int32, 0, 4) {
          for (nn: int32, 0, 16) {
            for (oo: int32, 0, 16) {
              let cse_var_13: int32 = (o.inner*256)
              let cse_var_12: int32 = (nn*16)
              Conv_3: Buffer(Conv_2, float32, [25690112], [])[(((((((((blockIdx.x*12845056) + (threadIdx.y*3211264)) + (n.inner*1605632)) + (blockIdx.z*8192)) + (blockIdx.y*2048)) + (threadIdx.z*1024)) + cse_var_13) + cse_var_12) + oo)] = Conv.wmma.accumulator_1[((((n.inner*1024) + cse_var_13) + cse_var_12) + oo)]
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