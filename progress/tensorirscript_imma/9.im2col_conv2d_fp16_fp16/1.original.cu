#[version = "0.0.5"]
@main = primfn(a: handle, w: handle, conv: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [1, 56, 56, 64], []),
             W: Buffer(W_1: Pointer(global float16), float16, [3, 3, 64, 64], []),
             Conv: Buffer(Conv_1: Pointer(global float16), float16, [1, 3136, 64], [])}
  buffer_map = {a: A, w: W, conv: Conv} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    Apad = alloc_buffer(float16[1, 58, 58, 64])
    data_im2col = alloc_buffer(float16[1, 3136, 576])
    weight_flatten = alloc_buffer(float16[576, 64])
    data_im2colPad = alloc_buffer(float16[1, 3200, 640])
    weight_flattenPad = alloc_buffer(float16[640, 128])
    CPad = alloc_buffer(float16[1, 3200, 128])
     {
      for (n: int32, 0, 1) {
        for (h: int32, 0, 58) {
          for (w_1: int32, 0, 58) {
            for (i: int32, 0, 64) {
              block([1, 58, 58, 64], "Apad") as [v_n, v_h, v_w, v_i] {
                bind(v_n, n)
                bind(v_h, h)
                bind(v_w, w_1)
                bind(v_i, i)
                tir.reads([A[v_n, (v_h - 1), (v_w - 1), v_i]])
                tir.writes([Apad[v_n, v_h, v_w, v_i]])
                Apad[v_n, v_h, v_w, v_i] = @tir.if_then_else(((((1 <= v_h) && (v_h < 57)) && (1 <= v_w)) && (v_w < 57)), A[v_n, (v_h - 1), (v_w - 1), v_i], 0f16, dtype=float16)
            }
          }
        }
      }
      for (n_1: int32, 0, 1) {
        for (x: int32, 0, 3136) {
          for (y: int32, 0, 576) {
            block([1, 3136, 576], "data_im2col") as [v_n_1, v_x, v_y] {
              bind(v_n_1, n_1)
              bind(v_x, x)
              bind(v_y, y)
              tir.reads([Apad[v_n_1, (floordiv(v_y, 192) + floordiv(v_x, 56)), (floordiv(floormod(v_y, 192), 64) + floormod(v_x, 56)), floormod(v_y, 64)]])
              tir.writes([data_im2col[v_n_1, v_x, v_y]])
              data_im2col[v_n_1, v_x, v_y] = Apad[v_n_1, ((1*floordiv(v_x, 56)) + (1*floordiv(floordiv(v_y, 64), 3))), ((1*floormod(v_x, 56)) + (1*floormod(floordiv(v_y, 64), 3))), floormod(v_y, 64)]
          }
        }
      }
      for (x_1: int32, 0, 576) {
        for (y_1: int32, 0, 64) {
          block([576, 64], "weight_flatten") as [v_x_1, v_y_1] {
            bind(v_x_1, x_1)
            bind(v_y_1, y_1)
            tir.reads([W[floordiv(v_x_1, 192), floordiv(floormod(v_x_1, 192), 64), floormod(v_x_1, 64), v_y_1]])
            tir.writes([weight_flatten[v_x_1, v_y_1]])
            weight_flatten[v_x_1, v_y_1] = W[floordiv(floordiv(v_x_1, 64), 3), floormod(floordiv(v_x_1, 64), 3), floormod(v_x_1, 64), v_y_1]
        }
      }
      for (n_2: int32, 0, 1) {
        for (i_1: int32, 0, 3200) {
          for (k: int32, 0, 640) {
            block([1, 3200, 640], "data_im2colPad") as [vn, vi, vk] {
              bind(vn, n_2)
              bind(vi, i_1)
              bind(vk, k)
              tir.reads([data_im2col[vn, vi, vk]])
              tir.writes([data_im2colPad[vn, vi, vk]])
              data_im2colPad[vn, vi, vk] = @tir.if_then_else(((vi < 3136) && (vk < 576)), data_im2col[vn, vi, vk], 0f16, dtype=float16)
          }
        }
      }
      for (k_1: int32, 0, 640) {
        for (j: int32, 0, 128) {
          block([640, 128], "weight_flattenPad") as [vk_1, vj] {
            bind(vk_1, k_1)
            bind(vj, j)
            tir.reads([weight_flatten[vk_1, vj]])
            tir.writes([weight_flattenPad[vk_1, vj]])
            weight_flattenPad[vk_1, vj] = @tir.if_then_else(((vk_1 < 576) && (vj < 64)), weight_flatten[vk_1, vj], 0f16, dtype=float16)
        }
      }
      for (n_3: int32, 0, 1) {
        for (x_2: int32, 0, 3200) {
          for (y_2: int32, 0, 128) {
            for (k_2: int32, 0, 640) {
              block([1, 3200, 128, tir.reduce_axis(0, 640)], "Conv") as [v_n_2, v_x_2, v_y_2, v_k] {
                bind(v_n_2, n_3)
                bind(v_x_2, x_2)
                bind(v_y_2, y_2)
                bind(v_k, k_2)
                tir.reads([data_im2colPad[v_n_2, v_x_2, v_k], weight_flattenPad[v_k, v_y_2]])
                tir.writes([CPad[v_n_2, v_x_2, v_y_2]])
                with init() {
                  CPad[v_n_2, v_x_2, v_y_2] = 0f16
                }
                CPad[v_n_2, v_x_2, v_y_2] = (CPad[v_n_2, v_x_2, v_y_2] + (data_im2colPad[v_n_2, v_x_2, v_k]*weight_flattenPad[v_k, v_y_2]))
            }
          }
        }
      }
      for (n_4: int32, 0, 1) {
        for (i_2: int32, 0, 3136) {
          for (j_1: int32, 0, 64) {
            block([1, 3136, 64], "CPad") as [vn_1, vi_1, vj_1] {
              bind(vn_1, n_4)
              bind(vi_1, i_2)
              bind(vj_1, j_1)
              tir.reads([CPad[vn_1, vi_1, vj_1]])
              tir.writes([Conv[vn_1, vi_1, vj_1]])
              Conv[vn_1, vi_1, vj_1] = CPad[vn_1, vi_1, vj_1]
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