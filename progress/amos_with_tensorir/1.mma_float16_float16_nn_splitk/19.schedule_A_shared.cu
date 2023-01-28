#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [1024, 16384], []),
             B: Buffer(B_1: Pointer(global float16), float16, [16384, 1024], []),
             C: Buffer(C_1: Pointer(global float16), float16, [1024, 1024], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    TC = alloc_buffer(float16[1, 1024, 1024])
    A_global = alloc_buffer(float16[64, 1024, 16, 16])
    A_global_shared = alloc_buffer(float16[64, 1024, 16, 16])
    A_global_shared_warp = alloc_buffer(float16[64, 1024, 16, 16])
    B_global = alloc_buffer(float16[1024, 64, 16, 16])
    B_global_shared = alloc_buffer(float16[1024, 64, 16, 16])
    B_global_shared_warp = alloc_buffer(float16[1024, 64, 16, 16])
    TC_warp = alloc_buffer(float16[1, 64, 64, 16, 16])
     {
      for (ax0: int32, 0, 16384) {
        for (ax1: int32, 0, 1024) {
          block([16384, 1024], "B_global") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([B[v0, v1]])
            tir.writes([B_global[floordiv(v0, 16), floordiv(v1, 16), ((floormod(v0, 8)*2) + floordiv(floormod(v1, 16), 8)), ((floordiv(floormod(v0, 16), 8)*8) + floormod(v1, 8))]])
            B_global[floordiv(v0, 16), floordiv(v1, 16), ((floormod(v0, 8)*2) + floordiv(floormod(v1, 16), 8)), ((floordiv(floormod(v0, 16), 8)*8) + floormod(v1, 8))] = B[v0, v1]
        }
      }
      for (ax0_1: int32, 0, 1024) {
        for (ax1_1: int32, 0, 16384) {
          block([1024, 16384], "A_global") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([A[v0_1, v1_1]])
            tir.writes([A_global[floordiv(v0_1, 16), floordiv(v1_1, 16), ((floormod(v0_1, 8)*2) + floordiv(floormod(v1_1, 16), 8)), ((floordiv(floormod(v0_1, 16), 8)*8) + floormod(v1_1, 8))]])
            A_global[floordiv(v0_1, 16), floordiv(v1_1, 16), ((floormod(v0_1, 8)*2) + floordiv(floormod(v1_1, 16), 8)), ((floordiv(floormod(v0_1, 16), 8)*8) + floormod(v1_1, 8))] = A[v0_1, v1_1]
        }
      }
      for (sk: int32, 0, 1) "thread_binding" {
        for (i_0_0: int32, 0, 8) "thread_binding" {
          for (j_0_0: int32, 0, 4) "thread_binding" {
            for (i_0_1: int32, 0, 1) "thread_binding" {
              for (j_0_1: int32, 0, 4) "thread_binding" {
                for (k_0_0: int32, 0, 512) {
                  for (ax0_0_ax1_0_ax0_1_ax1_1_fused_0: int32, 0, 1) "thread_binding" {
                    for (ax0_0_ax1_0_ax0_1_ax1_1_fused_1: int32, 0, 4) "thread_binding" {
                      for (ax0_0_ax1_0_ax0_1_ax1_1_fused_2: int32, 0, 4) {
                        for (ax0_0_ax1_0_ax0_1_ax1_1_fused_3: int32, 0, 32) "thread_binding" {
                          for (ax0_0_ax1_0_ax0_1_ax1_1_fused_4: int32, 0, 8) "vectorized" {
                            block([1024, 16384], "A_global_shared") as [v0_2, v1_2] {
                              bind(v0_2, (((i_0_0*128) + (floordiv((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 512)*16)) + floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 256), 16)))
                              bind(v1_2, (((k_0_0*32) + (floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 512), 256)*16)) + floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 16)))
                              tir.reads([A_global[floordiv(v0_2, 16), floordiv(v1_2, 16), ((floormod(v0_2, 8)*2) + floordiv(floormod(v1_2, 16), 8)), ((floordiv(floormod(v0_2, 16), 8)*8) + floormod(v1_2, 8))]])
                              tir.writes([A_global_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
                              A_global_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)] = A_global[floordiv(v0_2, 16), floordiv(v1_2, 16), ((floormod(v0_2, 8)*2) + floordiv(floormod(v1_2, 16), 8)), ((floordiv(floormod(v0_2, 16), 8)*8) + floormod(v1_2, 8))]
                          }
                        }
                      }
                    }
                  }
                  for (ax0_0: int32, 0, 2) {
                    for (ax1_0: int32, 0, 16) {
                      for (ax0_1_1: int32, 0, 16) {
                        for (ax1_1_1: int32, 0, 16) {
                          block([16384, 1024], "B_global_shared") as [v0_3, v1_3] {
                            bind(v0_3, (((k_0_0*32) + (ax0_0*16)) + ax0_1_1))
                            bind(v1_3, (((j_0_0*256) + (ax1_0*16)) + ax1_1_1))
                            tir.reads([B_global[floordiv(v0_3, 16), floordiv(v1_3, 16), ((floormod(v0_3, 8)*2) + floordiv(floormod(v1_3, 16), 8)), ((floordiv(floormod(v0_3, 16), 8)*8) + floormod(v1_3, 8))]])
                            tir.writes([B_global_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
                            B_global_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)] = B_global[floordiv(v0_3, 16), floordiv(v1_3, 16), ((floormod(v0_3, 8)*2) + floordiv(floormod(v1_3, 16), 8)), ((floordiv(floormod(v0_3, 16), 8)*8) + floormod(v1_3, 8))]
                        }
                      }
                    }
                  }
                  for (k_0_1: int32, 0, 2) {
                    for (ax0_0_1: int32, 0, 8) {
                      for (ax1_0_1: int32, 0, 1) {
                        for (ax0_1_2: int32, 0, 16) {
                          for (ax1_1_2: int32, 0, 16) {
                            block([1024, 16384], "A_global_shared_warp") as [v0_4, v1_4] {
                              bind(v0_4, (((i_0_0*128) + (ax0_0_1*16)) + ax0_1_2))
                              bind(v1_4, ((((k_0_0*32) + (k_0_1*16)) + (ax1_0_1*16)) + ax1_1_2))
                              tir.reads([A_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
                              tir.writes([A_global_shared_warp[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
                              A_global_shared_warp[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)] = A_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]
                          }
                        }
                      }
                    }
                    for (ax0_0_2: int32, 0, 1) {
                      for (ax1_0_2: int32, 0, 4) {
                        for (ax0_1_3: int32, 0, 16) {
                          for (ax1_1_3: int32, 0, 16) {
                            block([16384, 1024], "B_global_shared_warp") as [v0_5, v1_5] {
                              bind(v0_5, ((((k_0_0*32) + (k_0_1*16)) + (ax0_0_2*16)) + ax0_1_3))
                              bind(v1_5, ((((j_0_0*256) + (j_0_1*64)) + (ax1_0_2*16)) + ax1_1_3))
                              tir.reads([B_global_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
                              tir.writes([B_global_shared_warp[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
                              B_global_shared_warp[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)] = B_global_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]
                          }
                        }
                      }
                    }
                    for (i_0_2: int32, 0, 8) {
                      for (j_0_2: int32, 0, 4) {
                        for (i_1: int32, 0, 16) {
                          for (j_1: int32, 0, 16) {
                            for (k_1: int32, 0, 16) {
                              block([1, 1024, 1024, tir.reduce_axis(0, 16384)], "B") as [vsk, vi, vj, vk] {
                                bind(vsk, sk)
                                bind(vi, ((((i_0_0*128) + (i_0_1*128)) + (i_0_2*16)) + i_1))
                                bind(vj, ((((j_0_0*256) + (j_0_1*64)) + (j_0_2*16)) + j_1))
                                bind(vk, (((k_0_0*32) + (k_0_1*16)) + k_1))
                                tir.reads([A_global_shared_warp[floordiv(vi, 16), floordiv(vk, 16), floormod(vi, 16), floormod(vk, 16)], B_global_shared_warp[floordiv(vk, 16), floordiv(vj, 16), floormod(vk, 16), floormod(vj, 16)]])
                                tir.writes([TC_warp[vsk, floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)]])
                                with init() {
                                  TC_warp[vsk, floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] = 0f16
                                }
                                TC_warp[vsk, floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] = (TC_warp[vsk, floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] + (A_global_shared_warp[floordiv(vi, 16), floordiv(vk, 16), floormod(vi, 16), floormod(vk, 16)]*B_global_shared_warp[floordiv(vk, 16), floordiv(vj, 16), floormod(vk, 16), floormod(vj, 16)]))
                            }
                          }
                        }
                      }
                    }
                  }
                }
                for (ax0_0_3: int32, 0, 8) {
                  for (ax1_0_3: int32, 0, 4) {
                    for (ax0_1_4: int32, 0, 16) {
                      for (ax1_1_4: int32, 0, 16) {
                        block([1, 1024, 1024], "TC_warp") as [v0_6, v1_6, v2] {
                          bind(v0_6, 0)
                          bind(v1_6, (((i_0_0*128) + (ax0_0_3*16)) + ax0_1_4))
                          bind(v2, ((((j_0_0*256) + (j_0_1*64)) + (ax1_0_3*16)) + ax1_1_4))
                          tir.reads([TC_warp[v0_6, floordiv(v1_6, 16), floordiv(v2, 16), floormod(v1_6, 16), floormod(v2, 16)]])
                          tir.writes([TC[v0_6, v1_6, v2]])
                          TC[v0_6, v1_6, v2] = TC_warp[v0_6, floordiv(v1_6, 16), floordiv(v2, 16), floormod(v1_6, 16), floormod(v2, 16)]
                      }
                    }
                  }
                  for (ax0_2: int32, 0, 16) {
                    for (ax1_2: int32, 0, 64) {
                      block([1, 1024, 1024], "C") as [vsk_1, vi_1, vj_1] {
                        bind(vsk_1, 0)
                        bind(vi_1, (((i_0_0*128) + (ax0_0_3*16)) + ax0_2))
                        bind(vj_1, (((j_0_0*256) + (j_0_1*64)) + ax1_2))
                        tir.reads([C[vi_1, vj_1], TC[vsk_1, vi_1, vj_1]])
                        tir.writes([])
                        @tir.atomic_add(@tir.address_of(C[vi_1, vj_1], dtype=handle), TC[vsk_1, vi_1, vj_1], dtype=float16)
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