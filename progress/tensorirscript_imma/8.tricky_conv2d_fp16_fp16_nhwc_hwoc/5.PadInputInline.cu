#[version = "0.0.5"]
@main = primfn(var_A: handle, var_W: handle, var_Conv: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [16, 14, 14, 16, 16, 16], []),
             W: Buffer(W_1: Pointer(global float16), float16, [3, 3, 16, 32, 16, 16], []),
             Conv: Buffer(Conv_1: Pointer(global float16), float16, [16, 14, 14, 32, 16, 16], [])}
  buffer_map = {var_A: A, var_W: W, var_Conv: Conv} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    Apad = alloc_buffer(float16[16, 16, 16, 16, 16, 16])
    Apad_shared = alloc_buffer(float16[16, 16, 16, 16, 16, 16])
    Apad_shared_wmma.matrix_a = alloc_buffer(float16[16, 16, 16, 16, 16, 16])
    W_shared = alloc_buffer(float16[3, 3, 16, 32, 16, 16])
    W_shared_wmma.matrix_b = alloc_buffer(float16[3, 3, 16, 32, 16, 16])
    Conv_wmma.accumulator = alloc_buffer(float16[16, 14, 14, 32, 16, 16])
     {
      for (ax0: int32, 0, 3) {
        for (ax1: int32, 0, 3) {
          for (ax2: int32, 0, 16) {
            for (ax3: int32, 0, 32) {
              for (ax4: int32, 0, 16) {
                for (ax5: int32, 0, 16) {
                  block([3, 3, 16, 32, 16, 16], "W_shared") as [v0, v1, v2, v3, v4, v5] {
                    bind(v0, ax0)
                    bind(v1, ax1)
                    bind(v2, ax2)
                    bind(v3, ax3)
                    bind(v4, ax4)
                    bind(v5, ax5)
                    tir.reads([W[v0, v1, v2, v3, v4, v5]])
                    tir.writes([W_shared[v0, v1, v2, v3, v4, v5]])
                    W_shared[v0, v1, v2, v3, v4, v5] = W[v0, v1, v2, v3, v4, v5]
                }
              }
            }
          }
        }
      }
      for (n: int32, 0, 16) {
        for (h: int32, 0, 16) {
          for (w: int32, 0, 16) {
            for (i: int32, 0, 16) {
              for (nn: int32, 0, 16) {
                for (ii: int32, 0, 16) {
                  block([16, 16, 16, 16, 16, 16], "Apad_pad_const") as [v_n, v_h, v_w, v_i, v_nn, v_ii] {
                    bind(v_n, n)
                    bind(v_h, h)
                    bind(v_w, w)
                    bind(v_i, i)
                    bind(v_nn, nn)
                    bind(v_ii, ii)
                    tir.reads([])
                    tir.writes([Apad[v_n, v_h, v_w, v_i, v_nn, v_ii]])
                    Apad[v_n, v_h, v_w, v_i, v_nn, v_ii] = 0f16
                }
              }
            }
          }
        }
      }
      for (n_1: int32, 0, 16) {
        for (h_1: int32, 0, 14) {
          for (w_1: int32, 0, 14) {
            for (i_1: int32, 0, 16) {
              for (nn_1: int32, 0, 16) {
                for (ii_1: int32, 0, 16) {
                  block([16, 14, 14, 16, 16, 16], "Apad") as [v_n_1, v_h_1, v_w_1, v_i_1, v_nn_1, v_ii_1] {
                    bind(v_n_1, n_1)
                    bind(v_h_1, h_1)
                    bind(v_w_1, w_1)
                    bind(v_i_1, i_1)
                    bind(v_nn_1, nn_1)
                    bind(v_ii_1, ii_1)
                    tir.reads([A[v_n_1, v_h_1, v_w_1, v_i_1, v_nn_1, v_ii_1]])
                    tir.writes([Apad[v_n_1, (v_h_1 + 1), (v_w_1 + 1), v_i_1, v_nn_1, v_ii_1]])
                    Apad[v_n_1, (v_h_1 + 1), (v_w_1 + 1), v_i_1, v_nn_1, v_ii_1] = A[v_n_1, v_h_1, v_w_1, v_i_1, v_nn_1, v_ii_1]
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
                  block([16, 16, 16, 16, 16, 16], "Apad_shared") as [v0_1, v1_1, v2_1, v3_1, v4_1, v5_1] {
                    bind(v0_1, ax0_1)
                    bind(v1_1, ax1_1)
                    bind(v2_1, ax2_1)
                    bind(v3_1, ax3_1)
                    bind(v4_1, ax4_1)
                    bind(v5_1, ax5_1)
                    tir.reads([Apad[v0_1, v1_1, v2_1, v3_1, v4_1, v5_1]])
                    tir.writes([Apad_shared[v0_1, v1_1, v2_1, v3_1, v4_1, v5_1]])
                    Apad_shared[v0_1, v1_1, v2_1, v3_1, v4_1, v5_1] = Apad[v0_1, v1_1, v2_1, v3_1, v4_1, v5_1]
                }
              }
            }
          }
        }
      }
      for (ax0_2: int32, 0, 16) {
        for (ax1_2: int32, 0, 16) {
          for (ax2_2: int32, 0, 16) {
            for (ax3_2: int32, 0, 16) {
              for (ax4_2: int32, 0, 16) {
                for (ax5_2: int32, 0, 16) {
                  block([16, 16, 16, 16, 16, 16], "Apad_shared_wmma.matrix_a") as [v0_2, v1_2, v2_2, v3_2, v4_2, v5_2] {
                    bind(v0_2, ax0_2)
                    bind(v1_2, ax1_2)
                    bind(v2_2, ax2_2)
                    bind(v3_2, ax3_2)
                    bind(v4_2, ax4_2)
                    bind(v5_2, ax5_2)
                    tir.reads([Apad_shared[v0_2, v1_2, v2_2, v3_2, v4_2, v5_2]])
                    tir.writes([Apad_shared_wmma.matrix_a[v0_2, v1_2, v2_2, v3_2, v4_2, v5_2]])
                    Apad_shared_wmma.matrix_a[v0_2, v1_2, v2_2, v3_2, v4_2, v5_2] = Apad_shared[v0_2, v1_2, v2_2, v3_2, v4_2, v5_2]
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
                  block([3, 3, 16, 32, 16, 16], "W_shared_wmma.matrix_b") as [v0_3, v1_3, v2_3, v3_3, v4_3, v5_3] {
                    bind(v0_3, ax0_3)
                    bind(v1_3, ax1_3)
                    bind(v2_3, ax2_3)
                    bind(v3_3, ax3_3)
                    bind(v4_3, ax4_3)
                    bind(v5_3, ax5_3)
                    tir.reads([W_shared[v0_3, v1_3, v2_3, v3_3, v4_3, v5_3]])
                    tir.writes([W_shared_wmma.matrix_b[v0_3, v1_3, v2_3, v3_3, v4_3, v5_3]])
                    W_shared_wmma.matrix_b[v0_3, v1_3, v2_3, v3_3, v4_3, v5_3] = W_shared[v0_3, v1_3, v2_3, v3_3, v4_3, v5_3]
                }
              }
            }
          }
        }
      }
      for (n_2: int32, 0, 16) {
        for (h_2: int32, 0, 14) {
          for (w_2: int32, 0, 14) {
            for (o: int32, 0, 32) {
              for (nn_2: int32, 0, 16) {
                for (oo: int32, 0, 16) {
                  for (ic: int32, 0, 16) {
                    for (kh: int32, 0, 3) {
                      for (kw: int32, 0, 3) {
                        for (ii_2: int32, 0, 16) {
                          block([16, 14, 14, 32, 16, 16, tir.reduce_axis(0, 16), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3), tir.reduce_axis(0, 16)], "Conv") as [v_n_2, v_h_2, v_w_2, v_o, v_nn_2, v_oo, v_ic, v_kh, v_kw, v_ii_2] {
                            bind(v_n_2, n_2)
                            bind(v_h_2, h_2)
                            bind(v_w_2, w_2)
                            bind(v_o, o)
                            bind(v_nn_2, nn_2)
                            bind(v_oo, oo)
                            bind(v_ic, ic)
                            bind(v_kh, kh)
                            bind(v_kw, kw)
                            bind(v_ii_2, ii_2)
                            tir.reads([Apad_shared_wmma.matrix_a[v_n_2, (v_h_2 + v_kh), (v_w_2 + v_kw), v_ic, v_nn_2, v_ii_2], W_shared_wmma.matrix_b[v_kh, v_kw, v_ic, v_o, v_ii_2, v_oo]])
                            tir.writes([Conv_wmma.accumulator[v_n_2, v_h_2, v_w_2, v_o, v_nn_2, v_oo]])
                            with init() {
                              Conv_wmma.accumulator[v_n_2, v_h_2, v_w_2, v_o, v_nn_2, v_oo] = 0f16
                            }
                            Conv_wmma.accumulator[v_n_2, v_h_2, v_w_2, v_o, v_nn_2, v_oo] = (Conv_wmma.accumulator[v_n_2, v_h_2, v_w_2, v_o, v_nn_2, v_oo] + (Apad_shared_wmma.matrix_a[v_n_2, (v_h_2 + v_kh), (v_w_2 + v_kw), v_ic, v_nn_2, v_ii_2]*W_shared_wmma.matrix_b[v_kh, v_kw, v_ic, v_o, v_ii_2, v_oo]))
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
      for (ax0_4: int32, 0, 16) {
        for (ax1_4: int32, 0, 14) {
          for (ax2_4: int32, 0, 14) {
            for (ax3_4: int32, 0, 32) {
              for (ax4_4: int32, 0, 16) {
                for (ax5_4: int32, 0, 16) {
                  block([16, 14, 14, 32, 16, 16], "Conv_wmma.accumulator") as [v0_4, v1_4, v2_4, v3_4, v4_4, v5_4] {
                    bind(v0_4, ax0_4)
                    bind(v1_4, ax1_4)
                    bind(v2_4, ax2_4)
                    bind(v3_4, ax3_4)
                    bind(v4_4, ax4_4)
                    bind(v5_4, ax5_4)
                    tir.reads([Conv_wmma.accumulator[v0_4, v1_4, v2_4, v3_4, v4_4, v5_4]])
                    tir.writes([Conv[v0_4, v1_4, v2_4, v3_4, v4_4, v5_4]])
                    Conv[v0_4, v1_4, v2_4, v3_4, v4_4, v5_4] = Conv_wmma.accumulator[v0_4, v1_4, v2_4, v3_4, v4_4, v5_4]
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
      "data": [3]
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