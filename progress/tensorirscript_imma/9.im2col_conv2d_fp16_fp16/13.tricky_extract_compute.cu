#[version = "0.0.5"]
@main = primfn(var_A: handle, var_W: handle, var_Conv: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [1, 224, 224, 256], []),
             W: Buffer(W_1: Pointer(global float16), float16, [7, 7, 256, 512], []),
             Conv: Buffer(Conv_1: Pointer(global float16), float16, [1, 48400, 512], [])}
  buffer_map = {var_A: A, var_W: W, var_Conv: Conv} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    data_im2col_global = alloc_buffer(float16[1, 3025, 784, 16, 16])
    data_im2col_global_shared = alloc_buffer(float16[1, 3025, 784, 16, 16])
    data_im2col_global_shared_wmma.matrix_a = alloc_buffer(float16[1, 3025, 784, 16, 16])
    weight_flatten_global = alloc_buffer(float16[784, 32, 16, 16])
    weight_flatten_global_shared = alloc_buffer(float16[784, 32, 16, 16])
    weight_flatten_global_shared_wmma.matrix_b = alloc_buffer(float16[784, 32, 16, 16])
    Conv_wmma.accumulator = alloc_buffer(float16[1, 3025, 32, 16, 16])
     {
      for (ax0: int32, 0, 1) {
        for (ax1: int32, 0, 48400) {
          for (ax2: int32, 0, 12544) {
            block([1, 48400, 12544], "data_im2col_global") as [v0, v1, v2] {
              bind(v0, ax0)
              bind(v1, ax1)
              bind(v2, ax2)
              tir.reads([A[v0, ((floordiv(v2, 1792) + floordiv(v1, 220)) - 1), ((floordiv(floormod(v2, 1792), 256) + floormod(v1, 220)) - 1), floormod(v2, 256)]])
              tir.writes([data_im2col_global[v0, floordiv(v1, 16), floordiv(v2, 16), floormod(v1, 16), floormod(v2, 16)]])
              data_im2col_global[v0, floordiv(v1, 16), floordiv(v2, 16), floormod(v1, 16), floormod(v2, 16)] = @tir.if_then_else(((((1 <= (floordiv(v2, 1792) + floordiv(v1, 220))) && ((floordiv(v2, 1792) + floordiv(v1, 220)) < 225)) && (1 <= (floordiv(floormod(v2, 1792), 256) + floormod(v1, 220)))) && ((floordiv(floormod(v2, 1792), 256) + floormod(v1, 220)) < 225)), A[v0, ((floordiv(v2, 1792) + floordiv(v1, 220)) - 1), ((floordiv(floormod(v2, 1792), 256) + floormod(v1, 220)) - 1), floormod(v2, 256)], 0f16, dtype=float16)
          }
        }
      }
      for (ax0_1: int32, 0, 1) {
        for (ax1_1: int32, 0, 48400) {
          for (ax2_1: int32, 0, 12544) {
            block([1, 48400, 12544], "data_im2col_global_shared") as [v0_1, v1_1, v2_1] {
              bind(v0_1, ax0_1)
              bind(v1_1, ax1_1)
              bind(v2_1, ax2_1)
              tir.reads([data_im2col_global[v0_1, floordiv(v1_1, 16), floordiv(v2_1, 16), floormod(v1_1, 16), floormod(v2_1, 16)]])
              tir.writes([data_im2col_global_shared[v0_1, floordiv(v1_1, 16), floordiv(v2_1, 16), floormod(v1_1, 16), floormod(v2_1, 16)]])
              data_im2col_global_shared[v0_1, floordiv(v1_1, 16), floordiv(v2_1, 16), floormod(v1_1, 16), floormod(v2_1, 16)] = data_im2col_global[v0_1, floordiv(v1_1, 16), floordiv(v2_1, 16), floormod(v1_1, 16), floormod(v2_1, 16)]
          }
        }
      }
      for (ax0_2: int32, 0, 1) {
        for (ax1_2: int32, 0, 48400) {
          for (ax2_2: int32, 0, 12544) {
            block([1, 48400, 12544], "data_im2col_global_shared_wmma.matrix_a") as [v0_2, v1_2, v2_2] {
              bind(v0_2, ax0_2)
              bind(v1_2, ax1_2)
              bind(v2_2, ax2_2)
              tir.reads([data_im2col_global_shared[v0_2, floordiv(v1_2, 16), floordiv(v2_2, 16), floormod(v1_2, 16), floormod(v2_2, 16)]])
              tir.writes([data_im2col_global_shared_wmma.matrix_a[v0_2, floordiv(v1_2, 16), floordiv(v2_2, 16), floormod(v1_2, 16), floormod(v2_2, 16)]])
              data_im2col_global_shared_wmma.matrix_a[v0_2, floordiv(v1_2, 16), floordiv(v2_2, 16), floormod(v1_2, 16), floormod(v2_2, 16)] = data_im2col_global_shared[v0_2, floordiv(v1_2, 16), floordiv(v2_2, 16), floormod(v1_2, 16), floormod(v2_2, 16)]
          }
        }
      }
      for (ax0_3: int32, 0, 12544) {
        for (ax1_3: int32, 0, 512) {
          block([12544, 512], "weight_flatten_global") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([W[floordiv(v0_3, 1792), floordiv(floormod(v0_3, 1792), 256), floormod(v0_3, 256), v1_3]])
            tir.writes([weight_flatten_global[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
            weight_flatten_global[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)] = W[floordiv(v0_3, 1792), floordiv(floormod(v0_3, 1792), 256), floormod(v0_3, 256), v1_3]
        }
      }
      for (ax0_4: int32, 0, 12544) {
        for (ax1_4: int32, 0, 512) {
          block([12544, 512], "weight_flatten_global_shared") as [v0_4, v1_4] {
            bind(v0_4, ax0_4)
            bind(v1_4, ax1_4)
            tir.reads([weight_flatten_global[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
            tir.writes([weight_flatten_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
            weight_flatten_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)] = weight_flatten_global[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]
        }
      }
      for (ax0_5: int32, 0, 12544) {
        for (ax1_5: int32, 0, 512) {
          block([12544, 512], "weight_flatten_global_shared_wmma.matrix_b") as [v0_5, v1_5] {
            bind(v0_5, ax0_5)
            bind(v1_5, ax1_5)
            tir.reads([weight_flatten_global_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
            tir.writes([weight_flatten_global_shared_wmma.matrix_b[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
            weight_flatten_global_shared_wmma.matrix_b[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)] = weight_flatten_global_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]
        }
      }
      for (n: int32, 0, 1) {
        for (x_0: int32, 0, 3025) {
          for (y_0: int32, 0, 32) {
            for (k_0: int32, 0, 784) {
              for (x_1: int32, 0, 16) {
                for (y_1: int32, 0, 16) {
                  for (k_1: int32, 0, 16) {
                    block([1, 48400, 512, tir.reduce_axis(0, 12544)], "Conv") as [v_n, v_x, v_y, v_k] {
                      bind(v_n, n)
                      bind(v_x, ((x_0*16) + x_1))
                      bind(v_y, ((y_0*16) + y_1))
                      bind(v_k, ((k_0*16) + k_1))
                      tir.reads([data_im2col_global_shared_wmma.matrix_a[v_n, floordiv(v_x, 16), floordiv(v_k, 16), floormod(v_x, 16), floormod(v_k, 16)], weight_flatten_global_shared_wmma.matrix_b[floordiv(v_k, 16), floordiv(v_y, 16), floormod(v_k, 16), floormod(v_y, 16)]])
                      tir.writes([Conv_wmma.accumulator[v_n, floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)]])
                      with init() {
                        Conv_wmma.accumulator[v_n, floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)] = 0f16
                      }
                      Conv_wmma.accumulator[v_n, floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)] = (Conv_wmma.accumulator[v_n, floordiv(v_x, 16), floordiv(v_y, 16), floormod(v_x, 16), floormod(v_y, 16)] + (data_im2col_global_shared_wmma.matrix_a[v_n, floordiv(v_x, 16), floordiv(v_k, 16), floormod(v_x, 16), floormod(v_k, 16)]*weight_flatten_global_shared_wmma.matrix_b[floordiv(v_k, 16), floordiv(v_y, 16), floormod(v_k, 16), floormod(v_y, 16)]))
                  }
                }
              }
            }
          }
        }
      }
      for (ax0_6: int32, 0, 1) {
        for (ax1_6: int32, 0, 48400) {
          for (ax2_3: int32, 0, 512) {
            block([1, 48400, 512], "Conv_wmma.accumulator") as [v0_6, v1_6, v2_3] {
              bind(v0_6, ax0_6)
              bind(v1_6, ax1_6)
              bind(v2_3, ax2_3)
              tir.reads([Conv_wmma.accumulator[v0_6, floordiv(v1_6, 16), floordiv(v2_3, 16), floormod(v1_6, 16), floormod(v2_3, 16)]])
              tir.writes([Conv[v0_6, v1_6, v2_3]])
              Conv[v0_6, v1_6, v2_3] = Conv_wmma.accumulator[v0_6, floordiv(v1_6, 16), floordiv(v2_3, 16), floormod(v1_6, 16), floormod(v2_3, 16)]
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