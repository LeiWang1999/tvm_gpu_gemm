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
     {
      for (n: int32, 0, 16) {
        for (h: int32, 0, 16) {
          for (w: int32, 0, 16) {
            for (i: int32, 0, 16) {
              for (nn: int32, 0, 16) {
                for (ii: int32, 0, 16) {
                  block([16, 16, 16, 16, 16, 16], "Apad") as [v_n, v_h, v_w, v_i, v_nn, v_ii] {
                    bind(v_n, n)
                    bind(v_h, h)
                    bind(v_w, w)
                    bind(v_i, i)
                    bind(v_nn, nn)
                    bind(v_ii, ii)
                    tir.reads([A[v_n, (v_h - 1), (v_w - 1), v_i, v_nn, v_ii]])
                    tir.writes([Apad[v_n, v_h, v_w, v_i, v_nn, v_ii]])
                    Apad[v_n, v_h, v_w, v_i, v_nn, v_ii] = @tir.if_then_else(((((1 <= v_h) && (v_h < 15)) && (1 <= v_w)) && (v_w < 15)), A[v_n, (v_h - 1), (v_w - 1), v_i, v_nn, v_ii], 0f16, dtype=float16)
                }
              }
            }
          }
        }
      }
      for (n_1: int32, 0, 16) {
        for (h_1: int32, 0, 14) {
          for (w_1: int32, 0, 14) {
            for (o: int32, 0, 32) {
              for (nn_1: int32, 0, 16) {
                for (oo: int32, 0, 16) {
                  for (ic: int32, 0, 16) {
                    for (kh: int32, 0, 3) {
                      for (kw: int32, 0, 3) {
                        for (ii_1: int32, 0, 16) {
                          block([16, 14, 14, 32, 16, 16, tir.reduce_axis(0, 16), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3), tir.reduce_axis(0, 16)], "Conv") as [v_n_1, v_h_1, v_w_1, v_o, v_nn_1, v_oo, v_ic, v_kh, v_kw, v_ii_1] {
                            bind(v_n_1, n_1)
                            bind(v_h_1, h_1)
                            bind(v_w_1, w_1)
                            bind(v_o, o)
                            bind(v_nn_1, nn_1)
                            bind(v_oo, oo)
                            bind(v_ic, ic)
                            bind(v_kh, kh)
                            bind(v_kw, kw)
                            bind(v_ii_1, ii_1)
                            tir.reads([Apad[v_n_1, (v_h_1 + v_kh), (v_w_1 + v_kw), v_ic, v_nn_1, v_ii_1], W[v_kh, v_kw, v_ic, v_o, v_ii_1, v_oo]])
                            tir.writes([Conv[v_n_1, v_h_1, v_w_1, v_o, v_nn_1, v_oo]])
                            with init() {
                              Conv[v_n_1, v_h_1, v_w_1, v_o, v_nn_1, v_oo] = 0f16
                            }
                            Conv[v_n_1, v_h_1, v_w_1, v_o, v_nn_1, v_oo] = (Conv[v_n_1, v_h_1, v_w_1, v_o, v_nn_1, v_oo] + (Apad[v_n_1, (v_h_1 + v_kh), (v_w_1 + v_kw), v_ic, v_nn_1, v_ii_1]*W[v_kh, v_kw, v_ic, v_o, v_ii_1, v_oo]))
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