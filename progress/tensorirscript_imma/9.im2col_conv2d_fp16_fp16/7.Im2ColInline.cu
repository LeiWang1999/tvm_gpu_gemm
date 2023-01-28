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
    weight_flatten = alloc_buffer(float16[576, 64])
    data_im2colPad = alloc_buffer(float16[1, 3200, 640])
    weight_flattenPad = alloc_buffer(float16[640, 128])
    CPad = alloc_buffer(float16[1, 3200, 128])
    data_im2colPad_shared = alloc_buffer(float16[1, 3200, 640])
    data_im2colPad_shared_wmma.matrix_a = alloc_buffer(float16[1, 3200, 640])
    weight_flattenPad_shared = alloc_buffer(float16[640, 128])
    weight_flattenPad_shared_wmma.matrix_b = alloc_buffer(float16[640, 128])
    CPad_shared = alloc_buffer(float16[1, 3200, 128])
    CPad_shared_wmma.accumulator = alloc_buffer(float16[1, 3200, 128])
     {
      for (x: int32, 0, 576) {
        for (y: int32, 0, 64) {
          block([576, 64], "weight_flatten") as [v_x, v_y] {
            bind(v_x, x)
            bind(v_y, y)
            tir.reads([W[floordiv(v_x, 192), floordiv(floormod(v_x, 192), 64), floormod(v_x, 64), v_y]])
            tir.writes([weight_flatten[v_x, v_y]])
            weight_flatten[v_x, v_y] = W[floordiv(floordiv(v_x, 64), 3), floormod(floordiv(v_x, 64), 3), floormod(v_x, 64), v_y]
        }
      }
      for (n: int32, 0, 1) {
        for (i: int32, 0, 3200) {
          for (k: int32, 0, 640) {
            block([1, 3200, 640], "data_im2colPad") as [vn, vi, vk] {
              bind(vn, n)
              bind(vi, i)
              bind(vk, k)
              tir.reads([A[vn, ((floordiv(vk, 192) + floordiv(vi, 56)) - 1), ((floordiv(floormod(vk, 192), 64) + floormod(vi, 56)) - 1), floormod(vk, 64)]])
              tir.writes([data_im2colPad[vn, vi, vk]])
              data_im2colPad[vn, vi, vk] = @tir.if_then_else(((vi < 3136) && (vk < 576)), @tir.if_then_else(((((1 <= ((1*floordiv(vi, 56)) + (1*floordiv(floordiv(vk, 64), 3)))) && (((1*floordiv(vi, 56)) + (1*floordiv(floordiv(vk, 64), 3))) < 57)) && (1 <= ((1*floormod(vi, 56)) + (1*floormod(floordiv(vk, 64), 3))))) && (((1*floormod(vi, 56)) + (1*floormod(floordiv(vk, 64), 3))) < 57)), A[vn, (((1*floordiv(vi, 56)) + (1*floordiv(floordiv(vk, 64), 3))) - 1), (((1*floormod(vi, 56)) + (1*floormod(floordiv(vk, 64), 3))) - 1), floormod(vk, 64)], 0f16, dtype=float16), 0f16, dtype=float16)
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
      for (ax0: int32, 0, 1) {
        for (ax1: int32, 0, 3200) {
          for (ax2: int32, 0, 640) {
            block([1, 3200, 640], "data_im2colPad_shared") as [v0, v1, v2] {
              bind(v0, ax0)
              bind(v1, ax1)
              bind(v2, ax2)
              tir.reads([data_im2colPad[v0, v1, v2]])
              tir.writes([data_im2colPad_shared[v0, v1, v2]])
              data_im2colPad_shared[v0, v1, v2] = data_im2colPad[v0, v1, v2]
          }
        }
      }
      for (ax0_1: int32, 0, 1) {
        for (ax1_1: int32, 0, 3200) {
          for (ax2_1: int32, 0, 640) {
            block([1, 3200, 640], "data_im2colPad_shared_wmma.matrix_a") as [v0_1, v1_1, v2_1] {
              bind(v0_1, ax0_1)
              bind(v1_1, ax1_1)
              bind(v2_1, ax2_1)
              tir.reads([data_im2colPad_shared[v0_1, v1_1, v2_1]])
              tir.writes([data_im2colPad_shared_wmma.matrix_a[v0_1, v1_1, v2_1]])
              data_im2colPad_shared_wmma.matrix_a[v0_1, v1_1, v2_1] = data_im2colPad_shared[v0_1, v1_1, v2_1]
          }
        }
      }
      for (ax0_2: int32, 0, 640) {
        for (ax1_2: int32, 0, 128) {
          block([640, 128], "weight_flattenPad_shared") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([weight_flattenPad[v0_2, v1_2]])
            tir.writes([weight_flattenPad_shared[v0_2, v1_2]])
            weight_flattenPad_shared[v0_2, v1_2] = weight_flattenPad[v0_2, v1_2]
        }
      }
      for (ax0_3: int32, 0, 640) {
        for (ax1_3: int32, 0, 128) {
          block([640, 128], "weight_flattenPad_shared_wmma.matrix_b") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([weight_flattenPad_shared[v0_3, v1_3]])
            tir.writes([weight_flattenPad_shared_wmma.matrix_b[v0_3, v1_3]])
            weight_flattenPad_shared_wmma.matrix_b[v0_3, v1_3] = weight_flattenPad_shared[v0_3, v1_3]
        }
      }
      for (n_1: int32, 0, 1) {
        for (x_1: int32, 0, 3200) {
          for (y_1: int32, 0, 128) {
            for (k_2: int32, 0, 640) {
              block([1, 3200, 128, tir.reduce_axis(0, 640)], "Conv") as [v_n, v_x_1, v_y_1, v_k] {
                bind(v_n, n_1)
                bind(v_x_1, x_1)
                bind(v_y_1, y_1)
                bind(v_k, k_2)
                tir.reads([data_im2colPad_shared_wmma.matrix_a[v_n, v_x_1, v_k], weight_flattenPad_shared_wmma.matrix_b[v_k, v_y_1]])
                tir.writes([CPad_shared_wmma.accumulator[v_n, v_x_1, v_y_1]])
                with init() {
                  CPad_shared_wmma.accumulator[v_n, v_x_1, v_y_1] = 0f16
                }
                CPad_shared_wmma.accumulator[v_n, v_x_1, v_y_1] = (CPad_shared_wmma.accumulator[v_n, v_x_1, v_y_1] + (data_im2colPad_shared_wmma.matrix_a[v_n, v_x_1, v_k]*weight_flattenPad_shared_wmma.matrix_b[v_k, v_y_1]))
            }
          }
        }
      }
      for (ax0_4: int32, 0, 1) {
        for (ax1_4: int32, 0, 3200) {
          for (ax2_2: int32, 0, 128) {
            block([1, 3200, 128], "CPad_shared_wmma.accumulator") as [v0_4, v1_4, v2_2] {
              bind(v0_4, ax0_4)
              bind(v1_4, ax1_4)
              bind(v2_2, ax2_2)
              tir.reads([CPad_shared_wmma.accumulator[v0_4, v1_4, v2_2]])
              tir.writes([CPad_shared[v0_4, v1_4, v2_2]])
              CPad_shared[v0_4, v1_4, v2_2] = CPad_shared_wmma.accumulator[v0_4, v1_4, v2_2]
          }
        }
      }
      for (ax0_5: int32, 0, 1) {
        for (ax1_5: int32, 0, 3200) {
          for (ax2_3: int32, 0, 128) {
            block([1, 3200, 128], "CPad_shared") as [v0_5, v1_5, v2_3] {
              bind(v0_5, ax0_5)
              bind(v1_5, ax1_5)
              bind(v2_3, ax2_3)
              tir.reads([CPad_shared[v0_5, v1_5, v2_3]])
              tir.writes([CPad[v0_5, v1_5, v2_3]])
              CPad[v0_5, v1_5, v2_3] = CPad_shared[v0_5, v1_5, v2_3]
          }
        }
      }
      for (n_2: int32, 0, 1) {
        for (i_1: int32, 0, 3136) {
          for (j_1: int32, 0, 64) {
            block([1, 3136, 64], "CPad") as [vn_1, vi_1, vj_1] {
              bind(vn_1, n_2)
              bind(vi_1, i_1)
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