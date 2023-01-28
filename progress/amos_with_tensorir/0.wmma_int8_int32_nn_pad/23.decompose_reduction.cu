#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [3136, 576], []),
             B: Buffer(B_1: Pointer(global int8), int8, [576, 64], []),
             C: Buffer(C_1: Pointer(global int32), int32, [3136, 64], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    APad_global = alloc_buffer(int8[200, 40, 16, 16])
    APad_global_shared = alloc_buffer(int8[200, 40, 16, 16])
    APad_global_shared_wmma.matrix_a = alloc_buffer(int8[200, 40, 16, 16])
    BPad_global = alloc_buffer(int8[40, 8, 16, 16])
    BPad_global_shared = alloc_buffer(int8[40, 8, 16, 16])
    BPad_global_shared_wmma.matrix_b = alloc_buffer(int8[40, 8, 16, 16])
    CPad_shared = alloc_buffer(int32[3200, 128])
    CPad_shared_wmma.accumulator = alloc_buffer(int32[200, 8, 16, 16])
     {
      for (ax0: int32, 0, 3200) {
        for (ax1: int32, 0, 640) {
          block([3200, 640], "APad_global") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([A[v0, v1]])
            tir.writes([APad_global[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)]])
            APad_global[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)] = @tir.if_then_else(((v0 < 3136) && (v1 < 576)), A[v0, v1], 0i8, dtype=int8)
        }
      }
      for (ax0_1: int32, 0, 640) {
        for (ax1_1: int32, 0, 128) {
          block([640, 128], "BPad_global") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([B[v0_1, v1_1]])
            tir.writes([BPad_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]])
            BPad_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)] = @tir.if_then_else(((v0_1 < 576) && (v1_1 < 64)), B[v0_1, v1_1], 0i8, dtype=int8)
        }
      }
      for (i_0_0: int32, 0, 25) "thread_binding" {
        for (j_0_0: int32, 0, 1) "thread_binding" {
          for (i_0_1: int32, 0, 1) "thread_binding" {
            for (j_0_1: int32, 0, 4) "thread_binding" {
              for (i_0_2_init: int32, 0, 8) {
                for (j_0_2_init: int32, 0, 2) {
                  for (i_1_init: int32, 0, 16) {
                    for (j_1_init: int32, 0, 16) {
                      block([3200, 128], "B_init") as [vi, vj] {
                        bind(vi, ((((i_0_0*128) + (i_0_1*128)) + (i_0_2_init*16)) + i_1_init))
                        bind(vj, ((((j_0_0*128) + (j_0_1*32)) + (j_0_2_init*16)) + j_1_init))
                        tir.reads([])
                        tir.writes([CPad_shared_wmma.accumulator[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)]])
                        CPad_shared_wmma.accumulator[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] = 0
                    }
                  }
                }
              }
              for (k_0_0: int32, 0, 20) {
                for (ax0_0_ax1_0_ax0_1_ax1_1_fused_0: int32, 0, 1) "thread_binding" {
                  for (ax0_0_ax1_0_ax0_1_ax1_1_fused_1: int32, 0, 4) "thread_binding" {
                    for (ax0_0_ax1_0_ax0_1_ax1_1_fused_2: int32, 0, 2) {
                      for (ax0_0_ax1_0_ax0_1_ax1_1_fused_3: int32, 0, 32) "thread_binding" {
                        for (ax0_0_ax1_0_ax0_1_ax1_1_fused_4: int32, 0, 16) "vectorized" {
                          block([3200, 640], "APad_global_shared") as [v0_2, v1_2] {
                            bind(v0_2, (((i_0_0*128) + (floordiv((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 512)*16)) + floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 256), 16)))
                            bind(v1_2, (((k_0_0*32) + (floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 512), 256)*16)) + floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 16)))
                            tir.reads([APad_global[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
                            tir.writes([APad_global_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
                            APad_global_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)] = APad_global[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]
                        }
                      }
                    }
                  }
                }
                for (ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1: int32, 0, 1) "thread_binding" {
                  for (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1: int32, 0, 4) "thread_binding" {
                    for (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1: int32, 0, 2) {
                      for (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1: int32, 0, 32) "thread_binding" {
                        for (ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1: int32, 0, 16) "vectorized" {
                          block([640, 128], "BPad_global_shared") as [v0_3, v1_3] {
                            bind(v0_3, (((k_0_0*32) + (floordiv((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 2048)*16)) + floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 256), 16)))
                            bind(v1_3, ((floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 2048), 256)*16) + floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 16)))
                            tir.reads([BPad_global[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
                            tir.writes([BPad_global_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
                            BPad_global_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)] = BPad_global[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]
                        }
                      }
                    }
                  }
                }
                for (k_0_1: int32, 0, 2) {
                  for (ax0_0: int32, 0, 8) {
                    for (ax1_0: int32, 0, 1) {
                      for (ax0_1_1: int32, 0, 16) {
                        for (ax1_1_1: int32, 0, 16) {
                          block([3200, 640], "APad_global_shared_wmma.matrix_a") as [v0_4, v1_4] {
                            bind(v0_4, (((i_0_0*128) + (ax0_0*16)) + ax0_1_1))
                            bind(v1_4, ((((k_0_0*32) + (k_0_1*16)) + (ax1_0*16)) + ax1_1_1))
                            tir.reads([APad_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
                            tir.writes([APad_global_shared_wmma.matrix_a[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
                            APad_global_shared_wmma.matrix_a[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)] = APad_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]
                        }
                      }
                    }
                  }
                  for (ax0_0_1: int32, 0, 1) {
                    for (ax1_0_1: int32, 0, 2) {
                      for (ax0_1_2: int32, 0, 16) {
                        for (ax1_1_2: int32, 0, 16) {
                          block([640, 128], "BPad_global_shared_wmma.matrix_b") as [v0_5, v1_5] {
                            bind(v0_5, ((((k_0_0*32) + (k_0_1*16)) + (ax0_0_1*16)) + ax0_1_2))
                            bind(v1_5, (((j_0_1*32) + (ax1_0_1*16)) + ax1_1_2))
                            tir.reads([BPad_global_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
                            tir.writes([BPad_global_shared_wmma.matrix_b[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
                            BPad_global_shared_wmma.matrix_b[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)] = BPad_global_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]
                        }
                      }
                    }
                  }
                  for (i_0_2: int32, 0, 8) {
                    for (j_0_2: int32, 0, 2) {
                      for (i_1: int32, 0, 16) {
                        for (j_1: int32, 0, 16) {
                          for (k_1: int32, 0, 16) {
                            block([3200, 128, tir.reduce_axis(0, 640)], "B_update") as [vi_1, vj_1, vk] {
                              bind(vi_1, ((((i_0_0*128) + (i_0_1*128)) + (i_0_2*16)) + i_1))
                              bind(vj_1, ((((j_0_0*128) + (j_0_1*32)) + (j_0_2*16)) + j_1))
                              bind(vk, (((k_0_0*32) + (k_0_1*16)) + k_1))
                              tir.reads([CPad_shared_wmma.accumulator[floordiv(vi_1, 16), floordiv(vj_1, 16), floormod(vi_1, 16), floormod(vj_1, 16)], APad_global_shared_wmma.matrix_a[floordiv(vi_1, 16), floordiv(vk, 16), floormod(vi_1, 16), floormod(vk, 16)], BPad_global_shared_wmma.matrix_b[floordiv(vk, 16), floordiv(vj_1, 16), floormod(vk, 16), floormod(vj_1, 16)]])
                              tir.writes([CPad_shared_wmma.accumulator[floordiv(vi_1, 16), floordiv(vj_1, 16), floormod(vi_1, 16), floormod(vj_1, 16)]])
                              CPad_shared_wmma.accumulator[floordiv(vi_1, 16), floordiv(vj_1, 16), floormod(vi_1, 16), floormod(vj_1, 16)] = (CPad_shared_wmma.accumulator[floordiv(vi_1, 16), floordiv(vj_1, 16), floormod(vi_1, 16), floormod(vj_1, 16)] + (cast(int32, APad_global_shared_wmma.matrix_a[floordiv(vi_1, 16), floordiv(vk, 16), floormod(vi_1, 16), floormod(vk, 16)])*cast(int32, BPad_global_shared_wmma.matrix_b[floordiv(vk, 16), floordiv(vj_1, 16), floormod(vk, 16), floormod(vj_1, 16)])))
                          }
                        }
                      }
                    }
                  }
                }
              }
              for (ax0_0_2: int32, 0, 8) {
                for (ax1_0_2: int32, 0, 2) {
                  for (ax0_1_3: int32, 0, 16) {
                    for (ax1_1_3: int32, 0, 16) {
                      block([3200, 128], "CPad_shared_wmma.accumulator") as [v0_6, v1_6] {
                        bind(v0_6, (((i_0_0*128) + (ax0_0_2*16)) + ax0_1_3))
                        bind(v1_6, (((j_0_1*32) + (ax1_0_2*16)) + ax1_1_3))
                        tir.reads([CPad_shared_wmma.accumulator[floordiv(v0_6, 16), floordiv(v1_6, 16), floormod(v0_6, 16), floormod(v1_6, 16)]])
                        tir.writes([CPad_shared[v0_6, v1_6]])
                        CPad_shared[v0_6, v1_6] = CPad_shared_wmma.accumulator[floordiv(v0_6, 16), floordiv(v1_6, 16), floormod(v0_6, 16), floormod(v1_6, 16)]
                    }
                  }
                }
                for (ax0_ax1_fused_0: int32, 0, 4) {
                  for (ax0_ax1_fused_1: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_fused_2: int32, 0, 1) "thread_binding" {
                      for (ax0_ax1_fused_3: int32, 0, 32) "thread_binding" {
                        block([3200, 128], "CPad_shared") as [v0_7, v1_7] {
                          bind(v0_7, (((i_0_0*128) + (ax0_0_2*16)) + floordiv(((((ax0_ax1_fused_0*128) + (ax0_ax1_fused_1*32)) + (ax0_ax1_fused_2*32)) + ax0_ax1_fused_3), 32)))
                          bind(v1_7, ((j_0_1*32) + floormod(((((ax0_ax1_fused_0*128) + (ax0_ax1_fused_1*32)) + (ax0_ax1_fused_2*32)) + ax0_ax1_fused_3), 32)))
                          tir.reads([CPad_shared[v0_7, v1_7]])
                          tir.writes([C[v0_7, v1_7]])
                          C[v0_7, v1_7] = CPad_shared[v0_7, v1_7]
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