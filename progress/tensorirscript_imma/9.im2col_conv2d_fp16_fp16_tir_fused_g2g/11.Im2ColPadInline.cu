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
    weight_flattenPad = alloc_buffer(float16[576, 64])
    CPad = alloc_buffer(float16[1, 3136, 64])
    data_im2colPad_shared = alloc_buffer(float16[1, 3136, 576])
    data_im2colPad_shared_wmma.matrix_a = alloc_buffer(float16[1, 3136, 576])
    weight_flattenPad_shared = alloc_buffer(float16[576, 64])
    weight_flattenPad_shared_wmma.matrix_b = alloc_buffer(float16[576, 64])
    CPad_shared = alloc_buffer(float16[1, 3136, 64])
    CPad_shared_wmma.accumulator = alloc_buffer(float16[1, 3136, 64])
     {
      for (k: int32, 0, 576) {
        for (j: int32, 0, 64) {
          block([576, 64], "weight_flattenPad") as [vk, vj] {
            bind(vk, k)
            bind(vj, j)
            tir.reads([W[floordiv(vk, 192), floordiv(floormod(vk, 192), 64), floormod(vk, 64), vj]])
            tir.writes([weight_flattenPad[vk, vj]])
            weight_flattenPad[vk, vj] = @tir.if_then_else(((vk < 576) && (vj < 64)), W[floordiv(floordiv(vk, 64), 3), floormod(floordiv(vk, 64), 3), floormod(vk, 64), vj], 0f16, dtype=float16)
        }
      }
      for (ax0: int32, 0, 1) {
        for (ax1: int32, 0, 3136) {
          for (ax2: int32, 0, 576) {
            block([1, 3136, 576], "data_im2colPad_shared") as [v0, v1, v2] {
              bind(v0, ax0)
              bind(v1, ax1)
              bind(v2, ax2)
              tir.reads([A[v0, ((floordiv(v2, 192) + floordiv(v1, 56)) - 1), ((floordiv(floormod(v2, 192), 64) + floormod(v1, 56)) - 1), floormod(v2, 64)]])
              tir.writes([data_im2colPad_shared[v0, v1, v2]])
              data_im2colPad_shared[v0, v1, v2] = @tir.if_then_else(((v1 < 3136) && (v2 < 576)), @tir.if_then_else(((((1 <= ((1*floordiv(v1, 56)) + (1*floordiv(floordiv(v2, 64), 3)))) && (((1*floordiv(v1, 56)) + (1*floordiv(floordiv(v2, 64), 3))) < 57)) && (1 <= ((1*floormod(v1, 56)) + (1*floormod(floordiv(v2, 64), 3))))) && (((1*floormod(v1, 56)) + (1*floormod(floordiv(v2, 64), 3))) < 57)), A[v0, (((1*floordiv(v1, 56)) + (1*floordiv(floordiv(v2, 64), 3))) - 1), (((1*floormod(v1, 56)) + (1*floormod(floordiv(v2, 64), 3))) - 1), floormod(v2, 64)], 0f16, dtype=float16), 0f16, dtype=float16)
          }
        }
      }
      for (ax0_1: int32, 0, 1) {
        for (ax1_1: int32, 0, 3136) {
          for (ax2_1: int32, 0, 576) {
            block([1, 3136, 576], "data_im2colPad_shared_wmma.matrix_a") as [v0_1, v1_1, v2_1] {
              bind(v0_1, ax0_1)
              bind(v1_1, ax1_1)
              bind(v2_1, ax2_1)
              tir.reads([data_im2colPad_shared[v0_1, v1_1, v2_1]])
              tir.writes([data_im2colPad_shared_wmma.matrix_a[v0_1, v1_1, v2_1]])
              data_im2colPad_shared_wmma.matrix_a[v0_1, v1_1, v2_1] = data_im2colPad_shared[v0_1, v1_1, v2_1]
          }
        }
      }
      for (ax0_2: int32, 0, 576) {
        for (ax1_2: int32, 0, 64) {
          block([576, 64], "weight_flattenPad_shared") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([weight_flattenPad[v0_2, v1_2]])
            tir.writes([weight_flattenPad_shared[v0_2, v1_2]])
            weight_flattenPad_shared[v0_2, v1_2] = weight_flattenPad[v0_2, v1_2]
        }
      }
      for (ax0_3: int32, 0, 576) {
        for (ax1_3: int32, 0, 64) {
          block([576, 64], "weight_flattenPad_shared_wmma.matrix_b") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([weight_flattenPad_shared[v0_3, v1_3]])
            tir.writes([weight_flattenPad_shared_wmma.matrix_b[v0_3, v1_3]])
            weight_flattenPad_shared_wmma.matrix_b[v0_3, v1_3] = weight_flattenPad_shared[v0_3, v1_3]
        }
      }
      for (n: int32, 0, 1) {
        for (x: int32, 0, 3136) {
          for (y: int32, 0, 64) {
            for (k_1: int32, 0, 576) {
              block([1, 3136, 64, tir.reduce_axis(0, 576)], "Conv") as [v_n, v_x, v_y, v_k] {
                bind(v_n, n)
                bind(v_x, x)
                bind(v_y, y)
                bind(v_k, k_1)
                tir.reads([data_im2colPad_shared_wmma.matrix_a[v_n, v_x, v_k], weight_flattenPad_shared_wmma.matrix_b[v_k, v_y]])
                tir.writes([CPad_shared_wmma.accumulator[v_n, v_x, v_y]])
                with init() {
                  CPad_shared_wmma.accumulator[v_n, v_x, v_y] = 0f16
                }
                CPad_shared_wmma.accumulator[v_n, v_x, v_y] = (CPad_shared_wmma.accumulator[v_n, v_x, v_y] + (data_im2colPad_shared_wmma.matrix_a[v_n, v_x, v_k]*weight_flattenPad_shared_wmma.matrix_b[v_k, v_y]))
            }
          }
        }
      }
      for (ax0_4: int32, 0, 1) {
        for (ax1_4: int32, 0, 3136) {
          for (ax2_2: int32, 0, 64) {
            block([1, 3136, 64], "CPad_shared_wmma.accumulator") as [v0_4, v1_4, v2_2] {
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
        for (ax1_5: int32, 0, 3136) {
          for (ax2_3: int32, 0, 64) {
            block([1, 3136, 64], "CPad_shared") as [v0_5, v1_5, v2_3] {
              bind(v0_5, ax0_5)
              bind(v1_5, ax1_5)
              bind(v2_3, ax2_3)
              tir.reads([CPad_shared[v0_5, v1_5, v2_3]])
              tir.writes([CPad[v0_5, v1_5, v2_3]])
              CPad[v0_5, v1_5, v2_3] = CPad_shared[v0_5, v1_5, v2_3]
          }
        }
      }
      for (n_1: int32, 0, 1) {
        for (i: int32, 0, 3136) {
          for (j_1: int32, 0, 64) {
            block([1, 3136, 64], "CPad") as [vn, vi, vj_1] {
              bind(vn, n_1)
              bind(vi, i)
              bind(vj_1, j_1)
              tir.reads([CPad[vn, vi, vj_1]])
              tir.writes([Conv[vn, vi, vj_1]])
              Conv[vn, vi, vj_1] = CPad[vn, vi, vj_1]
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