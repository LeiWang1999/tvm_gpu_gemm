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
      for (n_0_0: int32, 0, 2) "thread_binding" {
        for (o_0_0: int32, 0, 4) "thread_binding" {
          for (n_0_1: int32, 0, 4) "thread_binding" {
            for (h_2: int32, 0, 14) {
              for (w_2: int32, 0, 14) {
                for (o_0_1: int32, 0, 2) "thread_binding" {
                  for (n_1_init: int32, 0, 2) {
                    for (o_1_init: int32, 0, 4) {
                      for (nn_init: int32, 0, 16) {
                        for (oo_init: int32, 0, 16) {
                          block([16, 14, 14, 32, 16, 16], "Conv_init") as [v_n_2, v_h_2, v_w_2, v_o, v_nn_2, v_oo] {
                            bind(v_n_2, (((n_0_0*8) + (n_0_1*2)) + n_1_init))
                            bind(v_h_2, h_2)
                            bind(v_w_2, w_2)
                            bind(v_o, (((o_0_0*8) + (o_0_1*4)) + o_1_init))
                            bind(v_nn_2, nn_init)
                            bind(v_oo, oo_init)
                            tir.reads([])
                            tir.writes([Conv_wmma.accumulator[v_n_2, v_h_2, v_w_2, v_o, v_nn_2, v_oo]])
                            Conv_wmma.accumulator[v_n_2, v_h_2, v_w_2, v_o, v_nn_2, v_oo] = 0f16
                        }
                      }
                    }
                  }
                  for (ic_0: int32, 0, 8) {
                    for (kh: int32, 0, 3) {
                      for (ax0_1_0: int32, 0, 2) "thread_binding" {
                        for (ax0_0: int32, 0, 4) "thread_binding" {
                          for (ax3_ax4_fused_1: int32, 0, 32) "thread_binding" {
                            for (ax1: int32, 0, 3) {
                              for (ax2: int32, 0, 2) {
                                for (ax3_ax4_fused_0: int32, 0, 8) {
                                  for (ax0_1_1: int32, 0, 1) {
                                    block([16, 16, 16, 16, 16, 16], "Apad_shared") as [v0, v1, v2, v3, v4, v5] {
                                      bind(v0, (((ax0_1_1 + (n_0_0*8)) + (ax0_0*2)) + ax0_1_0))
                                      bind(v1, (h_2 + kh))
                                      bind(v2, (w_2 + ax1))
                                      bind(v3, ((ic_0*2) + ax2))
                                      bind(v4, floordiv(((ax3_ax4_fused_0*32) + ax3_ax4_fused_1), 16))
                                      bind(v5, floormod(((ax3_ax4_fused_0*32) + ax3_ax4_fused_1), 16))
                                      tir.reads([Apad[v0, v1, v2, v3, v4, v5]])
                                      tir.writes([Apad_shared[v0, v1, v2, v3, v4, v5]])
                                      Apad_shared[v0, v1, v2, v3, v4, v5] = Apad[v0, v1, v2, v3, v4, v5]
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                      for (ax0_ax1_ax2_fused_0: int32, 0, 4) "thread_binding" {
                        for (ax0_ax1_ax2_fused_1_0: int32, 0, 2) "thread_binding" {
                          for (ax0_ax1_ax2_fused_1_1: int32, 0, 6) {
                            for (ax3_ax4_fused_0_1: int32, 0, 8) {
                              for (ax3_ax4_fused_1_1: int32, 0, 32) "thread_binding" {
                                block([3, 3, 16, 32, 16, 16], "W_shared") as [v0_1, v1_1, v2_1, v3_1, v4_1, v5_1] {
                                  bind(v0_1, kh)
                                  bind(v1_1, floordiv((((ax0_ax1_ax2_fused_0*12) + (ax0_ax1_ax2_fused_1_0*6)) + ax0_ax1_ax2_fused_1_1), 16))
                                  bind(v2_1, ((ic_0*2) + floordiv(floormod((((ax0_ax1_ax2_fused_0*12) + (ax0_ax1_ax2_fused_1_0*6)) + ax0_ax1_ax2_fused_1_1), 16), 8)))
                                  bind(v3_1, ((o_0_0*8) + floormod((((ax0_ax1_ax2_fused_0*12) + (ax0_ax1_ax2_fused_1_0*6)) + ax0_ax1_ax2_fused_1_1), 8)))
                                  bind(v4_1, floordiv(((ax3_ax4_fused_0_1*32) + ax3_ax4_fused_1_1), 16))
                                  bind(v5_1, floormod(((ax3_ax4_fused_0_1*32) + ax3_ax4_fused_1_1), 16))
                                  tir.reads([W[v0_1, v1_1, v2_1, v3_1, v4_1, v5_1]])
                                  tir.writes([W_shared[v0_1, v1_1, v2_1, v3_1, v4_1, v5_1]])
                                  W_shared[v0_1, v1_1, v2_1, v3_1, v4_1, v5_1] = W[v0_1, v1_1, v2_1, v3_1, v4_1, v5_1]
                              }
                            }
                          }
                        }
                      }
                      for (ic_1: int32, 0, 2) {
                        for (kw: int32, 0, 3) {
                          for (ax0: int32, 0, 2) {
                            for (ax1_1: int32, 0, 16) {
                              for (ax2_1: int32, 0, 16) {
                                block([16, 16, 16, 16, 16, 16], "Apad_shared_wmma.matrix_a") as [v0_2, v1_2, v2_2, v3_2, v4_2, v5_2] {
                                  bind(v0_2, (((n_0_0*8) + (n_0_1*2)) + ax0))
                                  bind(v1_2, (h_2 + kh))
                                  bind(v2_2, (w_2 + kw))
                                  bind(v3_2, ((ic_0*2) + ic_1))
                                  bind(v4_2, ax1_1)
                                  bind(v5_2, ax2_1)
                                  tir.reads([Apad_shared[v0_2, v1_2, v2_2, v3_2, v4_2, v5_2]])
                                  tir.writes([Apad_shared_wmma.matrix_a[v0_2, v1_2, v2_2, v3_2, v4_2, v5_2]])
                                  Apad_shared_wmma.matrix_a[v0_2, v1_2, v2_2, v3_2, v4_2, v5_2] = Apad_shared[v0_2, v1_2, v2_2, v3_2, v4_2, v5_2]
                              }
                            }
                          }
                          for (ax0_1: int32, 0, 4) {
                            for (ax1_2: int32, 0, 16) {
                              for (ax2_2: int32, 0, 16) {
                                block([3, 3, 16, 32, 16, 16], "W_shared_wmma.matrix_b") as [v0_3, v1_3, v2_3, v3_3, v4_3, v5_3] {
                                  bind(v0_3, kh)
                                  bind(v1_3, kw)
                                  bind(v2_3, ((ic_0*2) + ic_1))
                                  bind(v3_3, (((o_0_0*8) + (o_0_1*4)) + ax0_1))
                                  bind(v4_3, ax1_2)
                                  bind(v5_3, ax2_2)
                                  tir.reads([W_shared[v0_3, v1_3, v2_3, v3_3, v4_3, v5_3]])
                                  tir.writes([W_shared_wmma.matrix_b[v0_3, v1_3, v2_3, v3_3, v4_3, v5_3]])
                                  W_shared_wmma.matrix_b[v0_3, v1_3, v2_3, v3_3, v4_3, v5_3] = W_shared[v0_3, v1_3, v2_3, v3_3, v4_3, v5_3]
                              }
                            }
                          }
                          for (n_1_1: int32, 0, 2) {
                            for (o_1: int32, 0, 4) {
                              for (nn_2: int32, 0, 16) {
                                for (oo: int32, 0, 16) {
                                  for (ii_2: int32, 0, 16) {
                                    block([16, 14, 14, 32, 16, 16, tir.reduce_axis(0, 16), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3), tir.reduce_axis(0, 16)], "Conv_update") as [v_n_3, v_h_3, v_w_3, v_o_1, v_nn_3, v_oo_1, v_ic, v_kh, v_kw, v_ii_2] {
                                      bind(v_n_3, (((n_0_0*8) + (n_0_1*2)) + n_1_1))
                                      bind(v_h_3, h_2)
                                      bind(v_w_3, w_2)
                                      bind(v_o_1, (((o_0_0*8) + (o_0_1*4)) + o_1))
                                      bind(v_nn_3, nn_2)
                                      bind(v_oo_1, oo)
                                      bind(v_ic, ((ic_0*2) + ic_1))
                                      bind(v_kh, kh)
                                      bind(v_kw, kw)
                                      bind(v_ii_2, ii_2)
                                      tir.reads([Conv_wmma.accumulator[v_n_3, v_h_3, v_w_3, v_o_1, v_nn_3, v_oo_1], Apad_shared_wmma.matrix_a[v_n_3, (v_h_3 + v_kh), (v_w_3 + v_kw), v_ic, v_nn_3, v_ii_2], W_shared_wmma.matrix_b[v_kh, v_kw, v_ic, v_o_1, v_ii_2, v_oo_1]])
                                      tir.writes([Conv_wmma.accumulator[v_n_3, v_h_3, v_w_3, v_o_1, v_nn_3, v_oo_1]])
                                      Conv_wmma.accumulator[v_n_3, v_h_3, v_w_3, v_o_1, v_nn_3, v_oo_1] = (Conv_wmma.accumulator[v_n_3, v_h_3, v_w_3, v_o_1, v_nn_3, v_oo_1] + (Apad_shared_wmma.matrix_a[v_n_3, (v_h_3 + v_kh), (v_w_3 + v_kw), v_ic, v_nn_3, v_ii_2]*W_shared_wmma.matrix_b[v_kh, v_kw, v_ic, v_o_1, v_ii_2, v_oo_1]))
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                  for (ax0_2: int32, 0, 2) {
                    for (ax1_3: int32, 0, 4) {
                      for (ax2_3: int32, 0, 16) {
                        for (ax3: int32, 0, 16) {
                          block([16, 14, 14, 32, 16, 16], "Conv_wmma.accumulator") as [v0_4, v1_4, v2_4, v3_4, v4_4, v5_4] {
                            bind(v0_4, (((n_0_0*8) + (n_0_1*2)) + ax0_2))
                            bind(v1_4, h_2)
                            bind(v2_4, w_2)
                            bind(v3_4, (((o_0_0*8) + (o_0_1*4)) + ax1_3))
                            bind(v4_4, ax2_3)
                            bind(v5_4, ax3)
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