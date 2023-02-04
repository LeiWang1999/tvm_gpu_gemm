#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global float16), float16, [16384, 16384], []),
             C: Buffer(C_1: Pointer(global float16), float16, [16384, 16384], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(float16[1024, 1024, 16, 16])
    A_shared_warp = alloc_buffer(float16[1024, 1024, 32, 8])
    B_shared = alloc_buffer(float16[1024, 1024, 16, 16])
    B_shared_warp = alloc_buffer(float16[1024, 1024, 32, 8])
    C_warp = alloc_buffer(float16[1024, 1024, 32, 8])
    for (i_0_0: int32, 0, 128) "thread_binding" {
      for (j_0_0_0: int32, 0, 4) "thread_binding" {
        for (j_0_0_1: int32, 0, 16) "thread_binding" {
          for (i_0_1: int32, 0, 4) "thread_binding" {
            for (j_0_1: int32, 0, 1) "thread_binding" {
              for (i_0_2_init: int32, 0, 2) {
                for (j_0_2_init: int32, 0, 16) {
                  block([1024, 1024], "B_init_o") as [vi_o, vj_o] {
                    bind(vi_o, (((i_0_0*8) + (i_0_1*2)) + i_0_2_init))
                    bind(vj_o, (((j_0_0_0*256) + (j_0_0_1*16)) + j_0_2_init))
                    tir.reads([])
                    tir.writes([C_warp[vi_o, vj_o, 0:32, 0:8]])
                    C_warp_1 = match_buffer(C_warp[vi_o, vj_o, 0:32, 0:8])
                    attr [IterVar(tx: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
                    @tir.mma_fill(8, C_warp_2: Pointer(warp float16), elem_offset: int32, dtype=float16)
                }
              }
              for (k_0_0: int32, 0, 512) {
                for (ax0_0_ax1_0_ax0_1_ax1_1_fused_0: int32, 0, 4) "thread_binding" {
                  for (ax0_0_ax1_0_ax0_1_ax1_1_fused_1: int32, 0, 1) "thread_binding" {
                    for (ax0_0_ax1_0_ax0_1_ax1_1_fused_2: int32, 0, 4) {
                      for (ax0_0_ax1_0_ax0_1_ax1_1_fused_3: int32, 0, 32) "thread_binding" {
                        for (ax0_0_ax1_0_ax0_1_ax1_1_fused_4: int32, 0, 8) "vectorized" {
                          block([16384, 16384], "A_shared") as [v0, v1] {
                            bind(v0, (((i_0_0*128) + (floordiv((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 512)*16)) + floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 256), 16)))
                            bind(v1, (((k_0_0*32) + (floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 512), 256)*16)) + floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 16)))
                            tir.reads([A[(((floordiv(v0, 16)*16) + (floormod(v0, 8)*2)) + floordiv(floormod(v1, 16), 8)), (((floordiv(v1, 16)*16) + (floordiv(floormod(v0, 16), 8)*8)) + floormod(v1, 8))]])
                            tir.writes([A_shared[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)]])
                            A_shared[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)] = A[(((floordiv(v0, 16)*16) + (floormod(v0, 8)*2)) + floordiv(floormod(v1, 16), 8)), (((floordiv(v1, 16)*16) + (floordiv(floormod(v0, 16), 8)*8)) + floormod(v1, 8))]
                        }
                      }
                    }
                  }
                }
                for (ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1: int32, 0, 4) "thread_binding" {
                  for (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1: int32, 0, 1) "thread_binding" {
                    for (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1: int32, 0, 8) {
                      for (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1: int32, 0, 32) "thread_binding" {
                        for (ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1: int32, 0, 8) "vectorized" {
                          block([16384, 16384], "B_shared") as [v0_1, v1_1] {
                            bind(v0_1, (((k_0_0*32) + (floordiv((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*2048) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*2048)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 4096)*16)) + floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*2048) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*2048)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 256), 16)))
                            bind(v1_1, ((((j_0_0_0*4096) + (j_0_0_1*256)) + (floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*2048) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*2048)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 4096), 256)*16)) + floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*2048) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*2048)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 16)))
                            tir.reads([B[(((floordiv(v0_1, 16)*16) + (floormod(v0_1, 8)*2)) + floordiv(floormod(v1_1, 16), 8)), (((floordiv(v1_1, 16)*16) + (floordiv(floormod(v0_1, 16), 8)*8)) + floormod(v1_1, 8))]])
                            tir.writes([B_shared[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]])
                            B_shared[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)] = B[(((floordiv(v0_1, 16)*16) + (floormod(v0_1, 8)*2)) + floordiv(floormod(v1_1, 16), 8)), (((floordiv(v1_1, 16)*16) + (floordiv(floormod(v0_1, 16), 8)*8)) + floormod(v1_1, 8))]
                        }
                      }
                    }
                  }
                }
                for (k_0_1: int32, 0, 2) {
                  for (ax0_0: int32, 0, 2) {
                    for (ax1_0: int32, 0, 1) {
                      for (ax0_1: int32, 0, 16) {
                        for (ax1_1: int32, 0, 16) {
                          block([16384, 16384], "A_shared_warp") as [v0_2, v1_2] {
                            bind(v0_2, ((((i_0_0*128) + (i_0_1*32)) + (ax0_0*16)) + ax0_1))
                            bind(v1_2, ((((k_0_0*32) + (k_0_1*16)) + (ax1_0*16)) + ax1_1))
                            tir.reads([A_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
                            tir.writes([A_shared_warp[floordiv(v0_2, 16), floordiv(v1_2, 16), ((floormod(v0_2, 16)*2) + floordiv(floormod(v1_2, 16), 8)), floormod(v1_2, 8)]])
                            A_shared_warp[floordiv(v0_2, 16), floordiv(v1_2, 16), ((floormod(v0_2, 16)*2) + floordiv(floormod(v1_2, 16), 8)), floormod(v1_2, 8)] = A_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]
                        }
                      }
                    }
                  }
                  for (ax0_0_1: int32, 0, 1) {
                    for (ax1_0_1: int32, 0, 16) {
                      for (ax0_1_1: int32, 0, 16) {
                        for (ax1_1_1: int32, 0, 16) {
                          block([16384, 16384], "B_shared_warp") as [v0_3, v1_3] {
                            bind(v0_3, ((((k_0_0*32) + (k_0_1*16)) + (ax0_0_1*16)) + ax0_1_1))
                            bind(v1_3, ((((j_0_0_0*4096) + (j_0_0_1*256)) + (ax1_0_1*16)) + ax1_1_1))
                            tir.reads([B_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
                            tir.writes([B_shared_warp[floordiv(v0_3, 16), floordiv(v1_3, 16), ((floormod(v0_3, 16)*2) + floordiv(floormod(v1_3, 16), 8)), floormod(v1_3, 8)]])
                            B_shared_warp[floordiv(v0_3, 16), floordiv(v1_3, 16), ((floormod(v0_3, 16)*2) + floordiv(floormod(v1_3, 16), 8)), floormod(v1_3, 8)] = B_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]
                        }
                      }
                    }
                  }
                  for (i_0_2: int32, 0, 2) {
                    for (j_0_2: int32, 0, 16) {
                      for (i_1: int32, 0, 16) {
                        for (j_1: int32, 0, 16) {
                          for (k_1: int32, 0, 16) {
                            block([16384, 16384, tir.reduce_axis(0, 16384)], "B_update") as [vi, vj, vk] {
                              bind(vi, ((((i_0_0*128) + (i_0_1*32)) + (i_0_2*16)) + i_1))
                              bind(vj, (((((j_0_0_0*4096) + (j_0_0_1*256)) + (j_0_1*256)) + (j_0_2*16)) + j_1))
                              bind(vk, (((k_0_0*32) + (k_0_1*16)) + k_1))
                              tir.reads([C_warp[floordiv(vi, 16), floordiv(vj, 16), ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(floormod(vj, 16), 8)*4) + (floordiv(floormod(vi, 16), 8)*2)) + floormod(vj, 2))], A_shared_warp[floordiv(vi, 16), floordiv(vk, 16), ((floormod(vi, 16)*2) + floordiv(floormod(vk, 16), 8)), floormod(vk, 8)], B_shared_warp[floordiv(vk, 16), floordiv(vj, 16), ((floormod(vk, 16)*2) + floordiv(floormod(vj, 16), 8)), floormod(vj, 8)]])
                              tir.writes([C_warp[floordiv(vi, 16), floordiv(vj, 16), ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(floormod(vj, 16), 8)*4) + (floordiv(floormod(vi, 16), 8)*2)) + floormod(vj, 2))]])
                              C_warp[floordiv(vi, 16), floordiv(vj, 16), ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(floormod(vj, 16), 8)*4) + (floordiv(floormod(vi, 16), 8)*2)) + floormod(vj, 2))] = (C_warp[floordiv(vi, 16), floordiv(vj, 16), ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(floormod(vj, 16), 8)*4) + (floordiv(floormod(vi, 16), 8)*2)) + floormod(vj, 2))] + (A_shared_warp[floordiv(vi, 16), floordiv(vk, 16), ((floormod(vi, 16)*2) + floordiv(floormod(vk, 16), 8)), floormod(vk, 8)]*B_shared_warp[floordiv(vk, 16), floordiv(vj, 16), ((floormod(vk, 16)*2) + floordiv(floormod(vj, 16), 8)), floormod(vj, 8)]))
                          }
                        }
                      }
                    }
                  }
                }
              }
              for (ax0_0_2: int32, 0, 2) {
                for (ax1_0_2: int32, 0, 16) {
                  for (ax0_1_2: int32, 0, 16) {
                    for (ax1_1_2: int32, 0, 16) {
                      block([16384, 16384], "C_warp") as [v0_4, v1_4] {
                        bind(v0_4, ((((i_0_0*128) + (i_0_1*32)) + (ax0_0_2*16)) + ax0_1_2))
                        bind(v1_4, ((((j_0_0_0*4096) + (j_0_0_1*256)) + (ax1_0_2*16)) + ax1_1_2))
                        tir.reads([C_warp[floordiv(v0_4, 16), floordiv(v1_4, 16), ((floormod(v0_4, 8)*4) + floordiv(floormod(v1_4, 8), 2)), (((floordiv(floormod(v1_4, 16), 8)*4) + (floordiv(floormod(v0_4, 16), 8)*2)) + floormod(v1_4, 2))]])
                        tir.writes([C[v0_4, v1_4]])
                        C[v0_4, v1_4] = C_warp[floordiv(v0_4, 16), floordiv(v1_4, 16), ((floormod(v0_4, 8)*4) + floordiv(floormod(v1_4, 8), 2)), (((floordiv(floormod(v1_4, 16), 8)*4) + (floordiv(floormod(v0_4, 16), 8)*2)) + floormod(v1_4, 2))]
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