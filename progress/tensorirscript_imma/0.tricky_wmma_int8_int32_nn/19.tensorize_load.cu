#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [1024, 1024, 16, 16], []),
             B: Buffer(B_1: Pointer(global int8), int8, [1024, 1024, 16, 16], []),
             C: Buffer(C_1: Pointer(global int32), int32, [1024, 1024, 16, 16], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(int8[1024, 1024, 16, 16])
    A_shared_wmma.matrix_a = alloc_buffer(int8[1024, 1024, 16, 16])
    B_shared = alloc_buffer(int8[1024, 1024, 16, 16])
    B_shared_wmma.matrix_b = alloc_buffer(int8[1024, 1024, 16, 16])
    C_wmma.accumulator = alloc_buffer(int32[1024, 1024, 16, 16])
    for (ii_0: int32, 0, 64) "thread_binding" {
      for (jj_0_0: int32, 0, 16) "thread_binding" {
        for (jj_0_1: int32, 0, 16) "thread_binding" {
          for (ii_1: int32, 0, 1) "thread_binding" {
            for (jj_1: int32, 0, 4) "thread_binding" {
              for (ii_2_init: int32, 0, 16) {
                for (jj_2_init: int32, 0, 1) {
                  block([1024, 1024, 1, 1], "B_init_o") as [vii, vjj, vi_o, vj_o] {
                    bind(vii, (((ii_0*16) + (ii_1*16)) + ii_2_init))
                    bind(vjj, ((((jj_0_0*64) + (jj_0_1*4)) + jj_1) + jj_2_init))
                    bind(vi_o, 0)
                    bind(vj_o, 0)
                    tir.reads([])
                    tir.writes([C_wmma.accumulator[vii, vjj, 0:16, 0:16]])
                    C_2 = match_buffer(C_wmma.accumulator[vii, vjj, 0:16, 0:16])
                    @tir.tvm_fill_fragment(C_3: Pointer(wmma.accumulator int32), 16, 16, 16, ((floordiv(floordiv(elem_offset: int32, C_s0: int32), 16)*floordiv(C_s0, 16)) + floordiv(floormod(elem_offset, C_s0), 16)), 0f32, dtype=handle)
                }
              }
              for (kk_0: int32, 0, 512) {
                for (ax0_ax1_ax2_ax3_fused_0: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_ax3_fused_1: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_ax3_fused_2: int32, 0, 4) {
                      for (ax0_ax1_ax2_ax3_fused_3: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_ax3_fused_4: int32, 0, 16) "vectorized" {
                          block([1024, 1024, 16, 16], "A_shared") as [v0, v1, v2, v3] {
                            bind(v0, ((ii_0*16) + floordiv((((((ax0_ax1_ax2_ax3_fused_0*8192) + (ax0_ax1_ax2_ax3_fused_1*2048)) + (ax0_ax1_ax2_ax3_fused_2*512)) + (ax0_ax1_ax2_ax3_fused_3*16)) + ax0_ax1_ax2_ax3_fused_4), 512)))
                            bind(v1, ((kk_0*2) + floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0*8192) + (ax0_ax1_ax2_ax3_fused_1*2048)) + (ax0_ax1_ax2_ax3_fused_2*512)) + (ax0_ax1_ax2_ax3_fused_3*16)) + ax0_ax1_ax2_ax3_fused_4), 512), 256)))
                            bind(v2, floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0*8192) + (ax0_ax1_ax2_ax3_fused_1*2048)) + (ax0_ax1_ax2_ax3_fused_2*512)) + (ax0_ax1_ax2_ax3_fused_3*16)) + ax0_ax1_ax2_ax3_fused_4), 256), 16))
                            bind(v3, floormod((((((ax0_ax1_ax2_ax3_fused_0*8192) + (ax0_ax1_ax2_ax3_fused_1*2048)) + (ax0_ax1_ax2_ax3_fused_2*512)) + (ax0_ax1_ax2_ax3_fused_3*16)) + ax0_ax1_ax2_ax3_fused_4), 16))
                            tir.reads([A[v0, v1, v2, v3]])
                            tir.writes([A_shared[v0, v1, v2, v3]])
                            A_shared[v0, v1, v2, v3] = A[v0, v1, v2, v3]
                        }
                      }
                    }
                  }
                }
                for (ax0_ax1_ax2_ax3_fused_0_1: int32, 0, 1) "thread_binding" {
                  for (ax0_ax1_ax2_ax3_fused_1_1: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_ax2_ax3_fused_2_1: int32, 0, 1) {
                      for (ax0_ax1_ax2_ax3_fused_3_1: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_ax3_fused_4_1: int32, 0, 16) "vectorized" {
                          block([1024, 1024, 16, 16], "B_shared") as [v0_1, v1_1, v2_1, v3_1] {
                            bind(v0_1, ((kk_0*2) + floordiv((((((ax0_ax1_ax2_ax3_fused_0_1*2048) + (ax0_ax1_ax2_ax3_fused_1_1*512)) + (ax0_ax1_ax2_ax3_fused_2_1*512)) + (ax0_ax1_ax2_ax3_fused_3_1*16)) + ax0_ax1_ax2_ax3_fused_4_1), 1024)))
                            bind(v1_1, (((jj_0_0*64) + (jj_0_1*4)) + floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0_1*2048) + (ax0_ax1_ax2_ax3_fused_1_1*512)) + (ax0_ax1_ax2_ax3_fused_2_1*512)) + (ax0_ax1_ax2_ax3_fused_3_1*16)) + ax0_ax1_ax2_ax3_fused_4_1), 1024), 256)))
                            bind(v2_1, floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0_1*2048) + (ax0_ax1_ax2_ax3_fused_1_1*512)) + (ax0_ax1_ax2_ax3_fused_2_1*512)) + (ax0_ax1_ax2_ax3_fused_3_1*16)) + ax0_ax1_ax2_ax3_fused_4_1), 256), 16))
                            bind(v3_1, floormod((((((ax0_ax1_ax2_ax3_fused_0_1*2048) + (ax0_ax1_ax2_ax3_fused_1_1*512)) + (ax0_ax1_ax2_ax3_fused_2_1*512)) + (ax0_ax1_ax2_ax3_fused_3_1*16)) + ax0_ax1_ax2_ax3_fused_4_1), 16))
                            tir.reads([B[v0_1, v1_1, v2_1, v3_1]])
                            tir.writes([B_shared[v0_1, v1_1, v2_1, v3_1]])
                            B_shared[v0_1, v1_1, v2_1, v3_1] = B[v0_1, v1_1, v2_1, v3_1]
                        }
                      }
                    }
                  }
                }
                for (kk_1: int32, 0, 2) {
                  for (ax0: int32, 0, 16) {
                    block([1024, 1024, 1, 1], "A_shared_wmma.matrix_a_o") as [v0_2, v1_2, v2_o, v3_o] {
                      bind(v0_2, ((ii_0*16) + ax0))
                      bind(v1_2, ((kk_0*2) + kk_1))
                      bind(v2_o, 0)
                      bind(v3_o, 0)
                      tir.reads([A_shared[v0_2, v1_2, 0:16, 0:16]])
                      tir.writes([A_shared_wmma.matrix_a[v0_2, v1_2, 0:16, 0:16]])
                      A_2 = match_buffer(A_shared[v0_2, v1_2, 0:16, 0:16])
                      C_4 = match_buffer(A_shared_wmma.matrix_a[v0_2, v1_2, 0:16, 0:16])
                      @tir.tvm_load_matrix_sync(C_5: Pointer(wmma.matrix_a int8), 16, 16, 16, ((floordiv(floordiv(elem_offset_1: int32, C_s0_1: int32), 16)*floordiv(C_s0_1, 16)) + floordiv(floormod(elem_offset_1, C_s0_1), 16)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), A_3: Pointer(shared int8), elem_offset_2: int32, (A_s0: int32*16), 1, dtype=handle), A_s0, "row_major", dtype=handle)
                  }
                  for (ax0_1: int32, 0, 16) {
                    for (ax1: int32, 0, 16) {
                      block([1024, 1024, 16, 16], "B_shared_wmma.matrix_b") as [v0_3, v1_3, v2_2, v3_2] {
                        bind(v0_3, ((kk_0*2) + kk_1))
                        bind(v1_3, (((jj_0_0*64) + (jj_0_1*4)) + jj_1))
                        bind(v2_2, ax0_1)
                        bind(v3_2, ax1)
                        tir.reads([B_shared[v0_3, v1_3, v2_2, v3_2]])
                        tir.writes([B_shared_wmma.matrix_b[v0_3, v1_3, v2_2, v3_2]])
                        B_shared_wmma.matrix_b[v0_3, v1_3, v2_2, v3_2] = B_shared[v0_3, v1_3, v2_2, v3_2]
                    }
                  }
                  for (ii_2: int32, 0, 16) {
                    for (jj_2: int32, 0, 1) {
                      for (i: int32, 0, 16) {
                        for (j: int32, 0, 16) {
                          for (k: int32, 0, 16) {
                            block([1024, 1024, tir.reduce_axis(0, 1024), 16, 16, tir.reduce_axis(0, 16)], "B_update") as [vii_1, vjj_1, vkk, vi, vj, vk] {
                              bind(vii_1, (((ii_0*16) + (ii_1*16)) + ii_2))
                              bind(vjj_1, (((jj_2 + (jj_0_0*64)) + (jj_0_1*4)) + jj_1))
                              bind(vkk, ((kk_0*2) + kk_1))
                              bind(vi, i)
                              bind(vj, j)
                              bind(vk, k)
                              tir.reads([C_wmma.accumulator[vii_1, vjj_1, vi, vj], A_shared_wmma.matrix_a[vii_1, vkk, vi, vk], B_shared_wmma.matrix_b[vkk, vjj_1, vk, vj]])
                              tir.writes([C_wmma.accumulator[vii_1, vjj_1, vi, vj]])
                              C_wmma.accumulator[vii_1, vjj_1, vi, vj] = (C_wmma.accumulator[vii_1, vjj_1, vi, vj] + (cast(int32, A_shared_wmma.matrix_a[vii_1, vkk, vi, vk])*cast(int32, B_shared_wmma.matrix_b[vkk, vjj_1, vk, vj])))
                          }
                        }
                      }
                    }
                  }
                }
              }
              for (ax0_2: int32, 0, 16) {
                for (ax1_1: int32, 0, 16) {
                  for (ax2: int32, 0, 16) {
                    block([1024, 1024, 16, 16], "C_wmma.accumulator") as [v0_4, v1_4, v2_3, v3_3] {
                      bind(v0_4, ((ii_0*16) + ax0_2))
                      bind(v1_4, (((jj_0_0*64) + (jj_0_1*4)) + jj_1))
                      bind(v2_3, ax1_1)
                      bind(v3_3, ax2)
                      tir.reads([C_wmma.accumulator[v0_4, v1_4, v2_3, v3_3]])
                      tir.writes([C[v0_4, v1_4, v2_3, v3_3]])
                      C[v0_4, v1_4, v2_3, v3_3] = C_wmma.accumulator[v0_4, v1_4, v2_3, v3_3]
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