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
    A_shared_warp = alloc_buffer(int8[2048, 256, 8, 64])
    B_shared_warp = alloc_buffer(int8[2048, 256, 8, 64])
    C_warp = alloc_buffer(int32[16384, 16384])
    for (i_0: int32, 0, 128) "thread_binding" {
      for (j_0: int32, 0, 64) "thread_binding" {
        for (i_1_0: int32, 0, 2) "thread_binding" {
          for (j_1_0: int32, 0, 4) "thread_binding" {
            for (i_1_1_0_init: int32, 0, 4) {
              for (j_1_1_0_init: int32, 0, 4) {
                for (i_1_1_1_init: int32, 0, 16) {
                  for (j_1_1_1_init: int32, 0, 16) {
                    block([16384, 16384], "B_init") as [vi, vj] {
                      bind(vi, ((((i_0*128) + (i_1_0*64)) + (i_1_1_0_init*16)) + i_1_1_1_init))
                      bind(vj, ((((j_0*256) + (j_1_0*64)) + (j_1_1_0_init*16)) + j_1_1_1_init))
                      tir.reads([])
                      tir.writes([C_warp[vi, vj]])
                      C_warp[vi, vj] = 0
                  }
                }
              }
            }
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
              for (k_1_0: int32, 0, 2) {
                for (ax0: int32, 0, 8) {
                  for (ax1_0: int32, 0, 1) {
                    for (ax2_0: int32, 0, 2) {
                      for (ax1_1: int32, 0, 32) {
                        for (ax2_1: int32, 0, 16) {
                          block([2048, 256, 8, 64], "A_shared_warp") as [v0_2, v1_2, v2, v3] {
                            where((((ax1_0*32) + ax1_1) < 8))
                            bind(v0_2, (((i_0*16) + (i_1_0*8)) + ax0))
                            bind(v1_2, k_0)
                            bind(v2, ((ax1_0*32) + ax1_1))
                            bind(v3, (((k_1_0*32) + (ax2_0*16)) + ax2_1))
                            tir.reads([A_shared[v0_2, v1_2, v2, v3]])
                            tir.writes([A_shared_warp[v0_2, v1_2, v2, v3]])
                            A_shared_warp[v0_2, v1_2, v2, v3] = A_shared[v0_2, v1_2, v2, v3]
                        }
                      }
                    }
                  }
                }
                for (ax0_1: int32, 0, 8) {
                  for (ax1_0_1: int32, 0, 1) {
                    for (ax2_0_1: int32, 0, 2) {
                      for (ax1_1_1: int32, 0, 32) {
                        for (ax2_1_1: int32, 0, 16) {
                          block([2048, 256, 8, 64], "B_shared_warp") as [v0_3, v1_3, v2_1, v3_1] {
                            where((((ax1_0_1*32) + ax1_1_1) < 8))
                            bind(v0_3, (((j_0*32) + (j_1_0*8)) + ax0_1))
                            bind(v1_3, k_0)
                            bind(v2_1, ((ax1_0_1*32) + ax1_1_1))
                            bind(v3_1, (((k_1_0*32) + (ax2_0_1*16)) + ax2_1_1))
                            tir.reads([B_shared[v0_3, v1_3, v2_1, v3_1]])
                            tir.writes([B_shared_warp[v0_3, v1_3, v2_1, v3_1]])
                            B_shared_warp[v0_3, v1_3, v2_1, v3_1] = B_shared[v0_3, v1_3, v2_1, v3_1]
                        }
                      }
                    }
                  }
                }
                for (i_1_1_0: int32, 0, 4) {
                  for (j_1_1_0: int32, 0, 4) {
                    for (i_1_1_1: int32, 0, 16) {
                      for (j_1_1_1: int32, 0, 16) {
                        for (k_1_1: int32, 0, 32) {
                          block([16384, 16384, tir.reduce_axis(0, 16384)], "B_update") as [vi_1, vj_1, vk] {
                            bind(vi_1, ((((i_0*128) + (i_1_0*64)) + (i_1_1_0*16)) + i_1_1_1))
                            bind(vj_1, ((((j_0*256) + (j_1_0*64)) + (j_1_1_0*16)) + j_1_1_1))
                            bind(vk, (((k_0*64) + (k_1_0*32)) + k_1_1))
                            tir.reads([C_warp[vi_1, vj_1], A_shared_warp[floordiv(vi_1, 8), floordiv(vk, 64), floormod(vi_1, 8), floormod(vk, 64)], B_shared_warp[floordiv(vj_1, 8), floordiv(vk, 64), floormod(vj_1, 8), floormod(vk, 64)]])
                            tir.writes([C_warp[vi_1, vj_1]])
                            C_warp[vi_1, vj_1] = (C_warp[vi_1, vj_1] + (cast(int32, A_shared_warp[floordiv(vi_1, 8), floordiv(vk, 64), floormod(vi_1, 8), floormod(vk, 64)])*cast(int32, B_shared_warp[floordiv(vj_1, 8), floordiv(vk, 64), floormod(vj_1, 8), floormod(vk, 64)])))
                        }
                      }
                    }
                  }
                }
              }
            }
            for (ax0_0: int32, 0, 4) {
              for (ax1_0_2: int32, 0, 4) {
                for (ax0_1_1: int32, 0, 16) {
                  for (ax1_1_2: int32, 0, 16) {
                    block([16384, 16384], "C_warp") as [v0_4, v1_4] {
                      bind(v0_4, ((((i_0*128) + (i_1_0*64)) + (ax0_0*16)) + ax0_1_1))
                      bind(v1_4, ((((j_0*256) + (j_1_0*64)) + (ax1_0_2*16)) + ax1_1_2))
                      tir.reads([C_warp[v0_4, v1_4]])
                      tir.writes([C[v0_4, v1_4]])
                      C[v0_4, v1_4] = C_warp[v0_4, v1_4]
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