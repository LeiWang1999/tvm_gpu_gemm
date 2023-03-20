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
    data_im2col_shared = alloc_buffer(float16[14112, 64, 16, 16])
    data_im2col_shared_warp = alloc_buffer(float16[14112, 64, 16, 16])
    weight_flatten_shared = alloc_buffer(float16[24, 64, 16, 16])
    weight_flatten_shared_warp = alloc_buffer(float16[24, 64, 16, 16])
    Conv_warp = alloc_buffer(float16[14112, 24, 16, 16])
     {
      for (ax0: int32, 0, 225792) {
        for (ax1: int32, 0, 1024) {
          block([225792, 1024], "data_im2col_shared") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([A[floordiv(v0, 1764), (floordiv(v1, 1024) + floordiv(floormod(v0, 1764), 42)), floormod(v0, 42), floormod(v1, 1024)]])
            tir.writes([data_im2col_shared[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)]])
            data_im2col_shared[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)] = @tir.if_then_else(((((0 <= ((1*floordiv(floormod(v0, 1764), 42)) + (1*floordiv(floordiv(v1, 1024), 1)))) && (((1*floordiv(floormod(v0, 1764), 42)) + (1*floordiv(floordiv(v1, 1024), 1))) < 42)) && (0 <= ((1*floormod(floormod(v0, 1764), 42)) + (1*floormod(floordiv(v1, 1024), 1))))) && (((1*floormod(floormod(v0, 1764), 42)) + (1*floormod(floordiv(v1, 1024), 1))) < 42)), A[floordiv(v0, 1764), (((1*floordiv(floormod(v0, 1764), 42)) + (1*floordiv(floordiv(v1, 1024), 1))) - 0), (((1*floormod(floormod(v0, 1764), 42)) + (1*floormod(floordiv(v1, 1024), 1))) - 0), floormod(v1, 1024)], 0f16, dtype=float16)
        }
      }
      for (ax0_1: int32, 0, 225792) {
        for (ax1_1: int32, 0, 1024) {
          block([225792, 1024], "data_im2col_shared_warp") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([data_im2col_shared[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]])
            tir.writes([data_im2col_shared_warp[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]])
            data_im2col_shared_warp[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)] = data_im2col_shared[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]
        }
      }
      for (ax0_2: int32, 0, 384) {
        for (ax1_2: int32, 0, 1024) {
          block([384, 1024], "weight_flatten_shared") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([W[v0_2, floordiv(v1_2, 1024), 0, floormod(v1_2, 1024)]])
            tir.writes([weight_flatten_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
            weight_flatten_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)] = W[v0_2, floordiv(floordiv(v1_2, 1024), 1), floormod(floordiv(v1_2, 1024), 1), floormod(v1_2, 1024)]
        }
      }
      for (ax0_3: int32, 0, 384) {
        for (ax1_3: int32, 0, 1024) {
          block([384, 1024], "weight_flatten_shared_warp") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([weight_flatten_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
            tir.writes([weight_flatten_shared_warp[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
            weight_flatten_shared_warp[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)] = weight_flatten_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]
        }
      }
      for (x: int32, 0, 225792) {
        for (y: int32, 0, 384) {
          for (k: int32, 0, 1024) {
            block([225792, 384, tir.reduce_axis(0, 1024)], "Conv") as [v_x, v_y, v_k] {
              bind(v_x, x)
              bind(v_y, y)
              bind(v_k, k)
              tir.reads([data_im2col_shared_warp[floordiv(v_x, 16), floordiv(v_k, 16), floormod(v_x, 16), floormod(v_k, 16)], weight_flatten_shared_warp[floordiv(v_y, 16), floordiv(v_k, 16), floormod(v_y, 16), floormod(v_k, 16)]])
              tir.writes([Conv_warp[floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)]])
              with init() {
                Conv_warp[floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)] = 0f16
              }
              Conv_warp[floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)] = (Conv_warp[floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)] + (data_im2col_shared_warp[floordiv(v_x, 16), floordiv(v_k, 16), floormod(v_x, 16), floormod(v_k, 16)]*weight_flatten_shared_warp[floordiv(v_y, 16), floordiv(v_k, 16), floormod(v_y, 16), floormod(v_k, 16)]))
          }
        }
      }
      for (ax0_4: int32, 0, 225792) {
        for (ax1_4: int32, 0, 384) {
          block([225792, 384], "Conv_warp") as [v0_4, v1_4] {
            bind(v0_4, ax0_4)
            bind(v1_4, ax1_4)
            tir.reads([Conv_warp[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
            tir.writes([Conv[v0_4, v1_4]])
            Conv[v0_4, v1_4] = Conv_warp[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]
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