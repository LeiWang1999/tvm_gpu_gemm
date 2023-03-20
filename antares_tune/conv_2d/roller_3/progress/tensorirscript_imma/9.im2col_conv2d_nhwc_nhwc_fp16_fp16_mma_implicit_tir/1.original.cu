#[version = "0.0.5"]
@main = primfn(a: handle, w: handle, conv: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [128, 42, 42, 1024], []),
             W: Buffer(W_1: Pointer(global float16), float16, [384, 1, 1, 1024], []),
             Conv: Buffer(Conv_1: Pointer(global float16), float16, [225792, 384], [])}
  buffer_map = {a: A, w: W, conv: Conv} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    Apad = alloc_buffer(float16[128, 42, 42, 1024])
    data_im2col = alloc_buffer(float16[225792, 1024])
    weight_flatten = alloc_buffer(float16[384, 1024])
     {
      for (n: int32, 0, 128) {
        for (h: int32, 0, 42) {
          for (w_1: int32, 0, 42) {
            for (i: int32, 0, 1024) {
              block([128, 42, 42, 1024], "Apad") as [v_n, v_h, v_w, v_i] {
                bind(v_n, n)
                bind(v_h, h)
                bind(v_w, w_1)
                bind(v_i, i)
                tir.reads([A[v_n, v_h, v_w, v_i]])
                tir.writes([Apad[v_n, v_h, v_w, v_i]])
                Apad[v_n, v_h, v_w, v_i] = @tir.if_then_else(((((0 <= v_h) && (v_h < 42)) && (0 <= v_w)) && (v_w < 42)), A[v_n, (v_h - 0), (v_w - 0), v_i], 0f16, dtype=float16)
            }
          }
        }
      }
      for (x: int32, 0, 225792) {
        for (y: int32, 0, 1024) {
          block([225792, 1024], "data_im2col") as [v_x, v_y] {
            bind(v_x, x)
            bind(v_y, y)
            tir.reads([Apad[floordiv(v_x, 1764), (floordiv(v_y, 1024) + floordiv(floormod(v_x, 1764), 42)), floormod(v_x, 42), floormod(v_y, 1024)]])
            tir.writes([data_im2col[v_x, v_y]])
            data_im2col[v_x, v_y] = Apad[floordiv(v_x, 1764), ((1*floordiv(floormod(v_x, 1764), 42)) + (1*floordiv(floordiv(v_y, 1024), 1))), ((1*floormod(floormod(v_x, 1764), 42)) + (1*floormod(floordiv(v_y, 1024), 1))), floormod(v_y, 1024)]
        }
      }
      for (x_1: int32, 0, 384) {
        for (y_1: int32, 0, 1024) {
          block([384, 1024], "weight_flatten") as [v_n_1, v_k] {
            bind(v_n_1, x_1)
            bind(v_k, y_1)
            tir.reads([W[v_n_1, floordiv(v_k, 1024), 0, floormod(v_k, 1024)]])
            tir.writes([weight_flatten[v_n_1, v_k]])
            weight_flatten[v_n_1, v_k] = W[v_n_1, floordiv(floordiv(v_k, 1024), 1), floormod(floordiv(v_k, 1024), 1), floormod(v_k, 1024)]
        }
      }
      for (x_2: int32, 0, 225792) {
        for (y_2: int32, 0, 384) {
          for (k: int32, 0, 1024) {
            block([225792, 384, tir.reduce_axis(0, 1024)], "Conv") as [v_x_1, v_y_1, v_k_1] {
              bind(v_x_1, x_2)
              bind(v_y_1, y_2)
              bind(v_k_1, k)
              tir.reads([data_im2col[v_x_1, v_k_1], weight_flatten[v_y_1, v_k_1]])
              tir.writes([Conv[v_x_1, v_y_1]])
              with init() {
                Conv[v_x_1, v_y_1] = 0f16
              }
              Conv[v_x_1, v_y_1] = (Conv[v_x_1, v_y_1] + (data_im2col[v_x_1, v_k_1]*weight_flatten[v_y_1, v_k_1]))
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