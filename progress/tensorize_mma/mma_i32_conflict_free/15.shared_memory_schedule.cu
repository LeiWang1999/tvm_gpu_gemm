#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global int8), int8, [16384, 16384], []),
             C: Buffer(C_1: Pointer(global int32), int32, [16384, 16384], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(int8[2048, 256, 8, 64])
    B_shared = alloc_buffer(int8[2048, 256, 8, 64])
    for (i_0: int32, 0, 128) "thread_binding" {
      for (j_0: int32, 0, 64) "thread_binding" {
        for (i_1_0: int32, 0, 2) "thread_binding" {
          for (j_1_0: int32, 0, 4) "thread_binding" {
            for (k_0: int32, 0, 256) {
              for (ax0_ax1_fused_0: int32, 0, 2) {
                for (ax0_ax1_fused_1: int32, 0, 2) "thread_binding" {
                  for (ax0_ax1_fused_2: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_fused_3: int32, 0, 32) "thread_binding" {
                      for (ax0_ax1_fused_4: int32, 0, 16) "vectorized" {
                        block([16384, 16384], "A_shared") as [v0, v1] {
                          bind(v0, ((i_0*128) + floordiv((((((ax0_ax1_fused_0*4096) + (ax0_ax1_fused_1*2048)) + (ax0_ax1_fused_2*512)) + (ax0_ax1_fused_3*16)) + ax0_ax1_fused_4), 64)))
                          bind(v1, ((k_0*64) + floormod((((((ax0_ax1_fused_0*4096) + (ax0_ax1_fused_1*2048)) + (ax0_ax1_fused_2*512)) + (ax0_ax1_fused_3*16)) + ax0_ax1_fused_4), 64)))
                          tir.reads([A[v0, v1]])
                          tir.writes([A_shared[floordiv(v0, 8), floordiv(v1, 64), floormod(v0, 8), floormod(v1, 64)]])
                          A_shared[floordiv(v0, 8), floordiv(v1, 64), floormod(v0, 8), floormod(v1, 64)] = A[v0, v1]
                      }
                    }
                  }
                }
              }
              for (ax0_ax1_fused_0_1: int32, 0, 4) {
                for (ax0_ax1_fused_1_1: int32, 0, 2) "thread_binding" {
                  for (ax0_ax1_fused_2_1: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_fused_3_1: int32, 0, 32) "thread_binding" {
                      for (ax0_ax1_fused_4_1: int32, 0, 16) "vectorized" {
                        block([16384, 16384], "B_shared") as [v0_1, v1_1] {
                          bind(v0_1, ((j_0*256) + floordiv((((((ax0_ax1_fused_0_1*4096) + (ax0_ax1_fused_1_1*2048)) + (ax0_ax1_fused_2_1*512)) + (ax0_ax1_fused_3_1*16)) + ax0_ax1_fused_4_1), 64)))
                          bind(v1_1, ((k_0*64) + floormod((((((ax0_ax1_fused_0_1*4096) + (ax0_ax1_fused_1_1*2048)) + (ax0_ax1_fused_2_1*512)) + (ax0_ax1_fused_3_1*16)) + ax0_ax1_fused_4_1), 64)))
                          tir.reads([B[v0_1, v1_1]])
                          tir.writes([B_shared[floordiv(v0_1, 8), floordiv(v1_1, 64), floormod(v0_1, 8), floormod(v1_1, 64)]])
                          B_shared[floordiv(v0_1, 8), floordiv(v1_1, 64), floormod(v0_1, 8), floormod(v1_1, 64)] = B[v0_1, v1_1]
                      }
                    }
                  }
                }
              }
              for (i_1_1: int32, 0, 64) {
                for (j_1_1: int32, 0, 64) {
                  for (k_1: int32, 0, 64) {
                    block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi, vj, vk] {
                      bind(vi, (((i_0*128) + (i_1_0*64)) + i_1_1))
                      bind(vj, (((j_0*256) + (j_1_0*64)) + j_1_1))
                      bind(vk, ((k_0*64) + k_1))
                      tir.reads([A_shared[floordiv(vi, 8), floordiv(vk, 64), floormod(vi, 8), floormod(vk, 64)], B_shared[floordiv(vj, 8), floordiv(vk, 64), floormod(vj, 8), floormod(vk, 64)]])
                      tir.writes([C[vi, vj]])
                      with init() {
                        C[vi, vj] = 0
                      }
                      C[vi, vj] = (C[vi, vj] + (cast(int32, A_shared[floordiv(vi, 8), floordiv(vk, 64), floormod(vi, 8), floormod(vk, 64)])*cast(int32, B_shared[floordiv(vj, 8), floordiv(vk, 64), floormod(vj, 8), floormod(vk, 64)])))
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