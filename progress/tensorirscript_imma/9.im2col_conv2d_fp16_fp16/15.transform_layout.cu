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
    data_im2colPad_shared = alloc_buffer(float16[1, 200, 40, 16, 16])
    data_im2colPad_shared_wmma.matrix_a = alloc_buffer(float16[1, 200, 40, 16, 16])
    weight_flattenPad_shared = alloc_buffer(float16[40, 8, 16, 16])
    weight_flattenPad_shared_wmma.matrix_b = alloc_buffer(float16[40, 8, 16, 16])
    CPad_shared = alloc_buffer(float16[1, 3200, 128])
    CPad_shared_wmma.accumulator = alloc_buffer(float16[1, 200, 8, 16, 16])
     {
      for (ax0: int32, 0, 1) {
        for (ax1: int32, 0, 3200) {
          for (ax2: int32, 0, 640) {
            block([1, 3200, 640], "data_im2colPad_shared") as [v0, v1, v2] {
              bind(v0, ax0)
              bind(v1, ax1)
              bind(v2, ax2)
              tir.reads([A[v0, ((floordiv(v2, 192) + floordiv(v1, 56)) - 1), ((floordiv(floormod(v2, 192), 64) + floormod(v1, 56)) - 1), floormod(v2, 64)]])
              tir.writes([data_im2colPad_shared[v0, floordiv(v1, 16), floordiv(v2, 16), floormod(v1, 16), floormod(v2, 16)]])
              data_im2colPad_shared[v0, floordiv(v1, 16), floordiv(v2, 16), floormod(v1, 16), floormod(v2, 16)] = @tir.if_then_else(((v1 < 3136) && (v2 < 576)), @tir.if_then_else(((((1 <= ((1*floordiv(v1, 56)) + (1*floordiv(floordiv(v2, 64), 3)))) && (((1*floordiv(v1, 56)) + (1*floordiv(floordiv(v2, 64), 3))) < 57)) && (1 <= ((1*floormod(v1, 56)) + (1*floormod(floordiv(v2, 64), 3))))) && (((1*floormod(v1, 56)) + (1*floormod(floordiv(v2, 64), 3))) < 57)), A[v0, (((1*floordiv(v1, 56)) + (1*floordiv(floordiv(v2, 64), 3))) - 1), (((1*floormod(v1, 56)) + (1*floormod(floordiv(v2, 64), 3))) - 1), floormod(v2, 64)], 0f16, dtype=float16), 0f16, dtype=float16)
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
              tir.reads([data_im2colPad_shared[v0_1, floordiv(v1_1, 16), floordiv(v2_1, 16), floormod(v1_1, 16), floormod(v2_1, 16)]])
              tir.writes([data_im2colPad_shared_wmma.matrix_a[v0_1, floordiv(v1_1, 16), floordiv(v2_1, 16), floormod(v1_1, 16), floormod(v2_1, 16)]])
              data_im2colPad_shared_wmma.matrix_a[v0_1, floordiv(v1_1, 16), floordiv(v2_1, 16), floormod(v1_1, 16), floormod(v2_1, 16)] = data_im2colPad_shared[v0_1, floordiv(v1_1, 16), floordiv(v2_1, 16), floormod(v1_1, 16), floormod(v2_1, 16)]
          }
        }
      }
      for (ax0_2: int32, 0, 640) {
        for (ax1_2: int32, 0, 128) {
          block([640, 128], "weight_flattenPad_shared") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([W[floordiv(v0_2, 192), floordiv(floormod(v0_2, 192), 64), floormod(v0_2, 64), v1_2]])
            tir.writes([weight_flattenPad_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
            weight_flattenPad_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)] = @tir.if_then_else(((v0_2 < 576) && (v1_2 < 64)), W[floordiv(floordiv(v0_2, 64), 3), floormod(floordiv(v0_2, 64), 3), floormod(v0_2, 64), v1_2], 0f16, dtype=float16)
        }
      }
      for (ax0_3: int32, 0, 640) {
        for (ax1_3: int32, 0, 128) {
          block([640, 128], "weight_flattenPad_shared_wmma.matrix_b") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([weight_flattenPad_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
            tir.writes([weight_flattenPad_shared_wmma.matrix_b[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
            weight_flattenPad_shared_wmma.matrix_b[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)] = weight_flattenPad_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]
        }
      }
      for (n: int32, 0, 1) {
        for (x: int32, 0, 3200) {
          for (y: int32, 0, 128) {
            for (k: int32, 0, 640) {
              block([1, 3200, 128, tir.reduce_axis(0, 640)], "Conv") as [v_n, v_x, v_y, v_k] {
                bind(v_n, n)
                bind(v_x, x)
                bind(v_y, y)
                bind(v_k, k)
                tir.reads([data_im2colPad_shared_wmma.matrix_a[v_n, floordiv(v_x, 16), floordiv(v_k, 16), floormod(v_x, 16), floormod(v_k, 16)], weight_flattenPad_shared_wmma.matrix_b[floordiv(v_k, 16), floordiv(v_y, 16), floormod(v_k, 16), floormod(v_y, 16)]])
                tir.writes([CPad_shared_wmma.accumulator[v_n, floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)]])
                with init() {
                  CPad_shared_wmma.accumulator[v_n, floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)] = 0f16
                }
                CPad_shared_wmma.accumulator[v_n, floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)] = (CPad_shared_wmma.accumulator[v_n, floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)] + (data_im2colPad_shared_wmma.matrix_a[v_n, floordiv(v_x, 16), floordiv(v_k, 16), floormod(v_x, 16), floormod(v_k, 16)]*weight_flattenPad_shared_wmma.matrix_b[floordiv(v_k, 16), floordiv(v_y, 16), floormod(v_k, 16), floormod(v_y, 16)]))
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
              tir.reads([CPad_shared_wmma.accumulator[v0_4, floordiv(v1_4, 16), floordiv(v2_2, 16), floormod(v1_4, 16), floormod(v2_2, 16)]])
              tir.writes([CPad_shared[v0_4, v1_4, v2_2]])
              CPad_shared[v0_4, v1_4, v2_2] = CPad_shared_wmma.accumulator[v0_4, floordiv(v1_4, 16), floordiv(v2_2, 16), floormod(v1_4, 16), floormod(v2_2, 16)]
          }
        }
      }
      for (ax0_5: int32, 0, 1) {
        for (ax1_5: int32, 0, 3200) {
          for (ax2_3: int32, 0, 128) {
            block([1, 3200, 128], "CPad_shared") as [v0_5, v1_5, v2_3] {
              where(((ax1_5 < 3136) && (ax2_3 < 64)))
              bind(v0_5, ax0_5)
              bind(v1_5, ax1_5)
              bind(v2_3, ax2_3)
              tir.reads([CPad_shared[v0_5, v1_5, v2_3]])
              tir.writes([Conv[v0_5, v1_5, v2_3]])
              Conv[v0_5, v1_5, v2_3] = CPad_shared[v0_5, v1_5, v2_3]
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