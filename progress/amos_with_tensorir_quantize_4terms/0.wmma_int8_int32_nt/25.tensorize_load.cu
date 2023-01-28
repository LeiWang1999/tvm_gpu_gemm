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
    PA = alloc_buffer(int32[16384])
    PB = alloc_buffer(int32[16384])
    QC = alloc_buffer(int32[16384, 16384])
    A_global = alloc_buffer(int8[1024, 1024, 16, 16])
    A_global_shared = alloc_buffer(int8[1024, 1024, 16, 16])
    A_global_shared_wmma.matrix_a = alloc_buffer(int8[1024, 1024, 16, 16])
    B_global = alloc_buffer(int8[1024, 1024, 16, 16])
    B_global_shared = alloc_buffer(int8[1024, 1024, 16, 16])
    B_global_shared_wmma.matrix_b = alloc_buffer(int8[1024, 1024, 16, 16])
    QC_wmma.accumulator = alloc_buffer(int32[1024, 1024, 16, 16])
     {
      for (ax0: int32, 0, 16384) {
        for (ax1: int32, 0, 16384) {
          block([16384, 16384], "B_global") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([B[v0, v1]])
            tir.writes([B_global[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)]])
            B_global[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)] = B[v0, v1]
        }
      }
      for (ax0_1: int32, 0, 16384) {
        for (ax1_1: int32, 0, 16384) {
          block([16384, 16384], "A_global") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([A[v0_1, v1_1]])
            tir.writes([A_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]])
            A_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)] = A[v0_1, v1_1]
        }
      }
      for (i_0_0: int32, 0, 64) "thread_binding" {
        for (j_0_0: int32, 0, 256) "thread_binding" {
          for (i_0_1: int32, 0, 4) "thread_binding" {
            for (j_0_1: int32, 0, 1) "thread_binding" {
              for (i_0_2_init: int32, 0, 4) {
                for (j_0_2_init: int32, 0, 4) {
                  block([1024, 1024], "B_init_o") as [vi_o, vj_o] {
                    bind(vi_o, (((i_0_0*16) + (i_0_1*4)) + i_0_2_init))
                    bind(vj_o, (((j_0_0*4) + (j_0_1*4)) + j_0_2_init))
                    tir.reads([])
                    tir.writes([QC_wmma.accumulator[vi_o, vj_o, 0:16, 0:16]])
                    C_2 = match_buffer(QC_wmma.accumulator[vi_o, vj_o, 0:16, 0:16])
                    @tir.tvm_fill_fragment(C_3: Pointer(wmma.accumulator int32), 16, 16, 16, ((floordiv(floordiv(elem_offset: int32, C_s0: int32), 16)*floordiv(C_s0, 16)) + floordiv(floormod(elem_offset, C_s0), 16)), 0f32, dtype=handle)
                }
              }
              for (k_0_0: int32, 0, 256) {
                for (ax0_0_ax1_0_ax0_1_ax1_1_fused_0: int32, 0, 4) "thread_binding" {
                  for (ax0_0_ax1_0_ax0_1_ax1_1_fused_1: int32, 0, 1) "thread_binding" {
                    for (ax0_0_ax1_0_ax0_1_ax1_1_fused_2: int32, 0, 131072) {
                      for (ax0_0_ax1_0_ax0_1_ax1_1_fused_3: int32, 0, 32) "thread_binding" {
                        for (ax0_0_ax1_0_ax0_1_ax1_1_fused_4: int32, 0, 16) "vectorized" {
                          block([16384, 16384], "A_global_shared") as [v0_2, v1_2] {
                            bind(v0_2, ((floordiv((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*67108864) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*67108864)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 262144)*16) + floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*67108864) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*67108864)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 256), 16)))
                            bind(v1_2, ((floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*67108864) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*67108864)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 262144), 256)*16) + floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*67108864) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*67108864)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 16)))
                            tir.reads([A_global[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
                            tir.writes([A_global_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
                            A_global_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)] = A_global[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]
                        }
                      }
                    }
                  }
                }
                for (ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1: int32, 0, 4) "thread_binding" {
                  for (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1: int32, 0, 1) "thread_binding" {
                    for (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1: int32, 0, 2) {
                      for (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1: int32, 0, 32) "thread_binding" {
                        for (ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1: int32, 0, 16) "vectorized" {
                          block([16384, 16384], "B_global_shared") as [v0_3, v1_3] {
                            bind(v0_3, (((j_0_0*64) + (floordiv((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 1024)*16)) + floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 256), 16)))
                            bind(v1_3, (((k_0_0*64) + (floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 1024), 256)*16)) + floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*512)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*16)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 16)))
                            tir.reads([B_global[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
                            tir.writes([B_global_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
                            B_global_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)] = B_global[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]
                        }
                      }
                    }
                  }
                }
                for (k_0_1: int32, 0, 4) {
                  for (ax0_0: int32, 0, 4) {
                    for (ax1_0: int32, 0, 1) {
                      block([1024, 1024], "A_global_shared_wmma.matrix_a_o") as [v0_o, v1_o] {
                        bind(v0_o, (((i_0_0*16) + (i_0_1*4)) + ax0_0))
                        bind(v1_o, (((k_0_0*4) + k_0_1) + ax1_0))
                        tir.reads([A_global_shared[v0_o, v1_o, 0:16, 0:16]])
                        tir.writes([A_global_shared_wmma.matrix_a[v0_o, v1_o, 0:16, 0:16]])
                        A_2 = match_buffer(A_global_shared[v0_o, v1_o, 0:16, 0:16])
                        C_4 = match_buffer(A_global_shared_wmma.matrix_a[v0_o, v1_o, 0:16, 0:16])
                        @tir.tvm_load_matrix_sync(C_5: Pointer(wmma.matrix_a int8), 16, 16, 16, ((floordiv(floordiv(elem_offset_1: int32, C_s0_1: int32), 16)*floordiv(C_s0_1, 16)) + floordiv(floormod(elem_offset_1, C_s0_1), 16)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), A_3: Pointer(shared int8), elem_offset_2: int32, (A_s0: int32*16), 1, dtype=handle), A_s0, "row_major", dtype=handle)
                    }
                  }
                  for (ax0_0_1: int32, 0, 4) {
                    for (ax1_0_1: int32, 0, 1) {
                      for (ax0_1_1: int32, 0, 16) {
                        for (ax1_1_1: int32, 0, 16) {
                          block([16384, 16384], "B_global_shared_wmma.matrix_b") as [v0_4, v1_4] {
                            bind(v0_4, (((j_0_0*64) + (ax0_0_1*16)) + ax0_1_1))
                            bind(v1_4, ((((k_0_0*64) + (k_0_1*16)) + (ax1_0_1*16)) + ax1_1_1))
                            tir.reads([B_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
                            tir.writes([B_global_shared_wmma.matrix_b[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
                            B_global_shared_wmma.matrix_b[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)] = B_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]
                        }
                      }
                    }
                  }
                  for (i_0_2: int32, 0, 4) {
                    for (j_0_2: int32, 0, 4) {
                      for (i_1: int32, 0, 16) {
                        for (j_1: int32, 0, 16) {
                          for (k_1: int32, 0, 16) {
                            block([16384, 16384, tir.reduce_axis(0, 16384)], "B_update") as [vi, vj, vk] {
                              bind(vi, ((((i_0_0*256) + (i_0_1*64)) + (i_0_2*16)) + i_1))
                              bind(vj, ((((j_0_0*64) + (j_0_1*64)) + (j_0_2*16)) + j_1))
                              bind(vk, (((k_0_0*64) + (k_0_1*16)) + k_1))
                              tir.reads([QC_wmma.accumulator[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)], A_global_shared_wmma.matrix_a[floordiv(vi, 16), floordiv(vk, 16), floormod(vi, 16), floormod(vk, 16)], B_global_shared_wmma.matrix_b[floordiv(vj, 16), floordiv(vk, 16), floormod(vj, 16), floormod(vk, 16)]])
                              tir.writes([QC_wmma.accumulator[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)]])
                              QC_wmma.accumulator[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] = (QC_wmma.accumulator[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] + (cast(int32, A_global_shared_wmma.matrix_a[floordiv(vi, 16), floordiv(vk, 16), floormod(vi, 16), floormod(vk, 16)])*cast(int32, B_global_shared_wmma.matrix_b[floordiv(vj, 16), floordiv(vk, 16), floormod(vj, 16), floormod(vk, 16)])))
                          }
                        }
                      }
                    }
                  }
                  for (ax0_2: int32, 0, 16384) {
                    for (ax1_2: int32, 0, 16384) {
                      block([16384, tir.reduce_axis(0, 16384)], "Pre_compute_A") as [vi_1, vk_1] {
                        bind(vi_1, ax0_2)
                        bind(vk_1, ax1_2)
                        tir.reads([A_global_shared[floordiv(vi_1, 16), floordiv(vk_1, 16), floormod(vi_1, 16), floormod(vk_1, 16)]])
                        tir.writes([PA[vi_1]])
                        with init() {
                          PA[vi_1] = 0
                        }
                        PA[vi_1] = (PA[vi_1] + (4*cast(int32, A_global_shared[floordiv(vi_1, 16), floordiv(vk_1, 16), floormod(vi_1, 16), floormod(vk_1, 16)])))
                    }
                  }
                }
              }
              for (ax0_0_2: int32, 0, 4) {
                for (ax1_0_2: int32, 0, 4) {
                  for (ax0_1_2: int32, 0, 16) {
                    for (ax1_1_2: int32, 0, 16) {
                      block([16384, 16384], "QC_wmma.accumulator") as [v0_5, v1_5] {
                        bind(v0_5, ((((i_0_0*256) + (i_0_1*64)) + (ax0_0_2*16)) + ax0_1_2))
                        bind(v1_5, (((j_0_0*64) + (ax1_0_2*16)) + ax1_1_2))
                        tir.reads([QC_wmma.accumulator[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
                        tir.writes([QC[v0_5, v1_5]])
                        QC[v0_5, v1_5] = QC_wmma.accumulator[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]
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