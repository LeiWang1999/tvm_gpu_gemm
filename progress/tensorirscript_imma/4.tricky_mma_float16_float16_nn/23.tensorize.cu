#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [196, 36, 16, 16], []),
             B: Buffer(B_1: Pointer(global float16), float16, [36, 4, 16, 16], []),
             C: Buffer(C_1: Pointer(global float16), float16, [1, 196, 4, 16, 16], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(float16[196, 36, 16, 16])
    A_shared_warp = alloc_buffer(float16[196, 36, 32, 8])
    B_shared = alloc_buffer(float16[36, 4, 16, 16])
    B_shared_warp = alloc_buffer(float16[36, 4, 32, 8])
    C_warp = alloc_buffer(float16[1, 196, 4, 32, 8])
    for (sk: int32, 0, 1) "thread_binding" {
      for (ii_0: int32, 0, 49) "thread_binding" {
        for (jj_0: int32, 0, 1) "thread_binding" {
          for (ii_1: int32, 0, 2) "thread_binding" {
            for (jj_1: int32, 0, 2) "thread_binding" {
              for (ii_2_init: int32, 0, 2) {
                for (jj_2_init: int32, 0, 2) {
                  block([1, 196, 4, 1, 1], "B_init_o") as [vsk, vii, vjj, vi_o, vj_o] {
                    bind(vsk, sk)
                    bind(vii, (((ii_0*4) + (ii_1*2)) + ii_2_init))
                    bind(vjj, (((jj_0*4) + (jj_1*2)) + jj_2_init))
                    bind(vi_o, 0)
                    bind(vj_o, 0)
                    tir.reads([])
                    tir.writes([C_warp[vsk, vii, vjj, 0:32, 0:8]])
                    C_warp_1 = match_buffer(C_warp[vsk, vii, vjj, 0:32, 0:8])
                    attr [IterVar(tx: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
                    @tir.mma_fill(8, C_warp_2: Pointer(warp float16), elem_offset: int32, dtype=float16)
                }
              }
              for (kk_0: int32, 0, 18) {
                for (ax0_ax1_ax2_ax3_fused_0: int32, 0, 2) "thread_binding" {
                  for (ax0_ax1_ax2_ax3_fused_1: int32, 0, 2) "thread_binding" {
                    for (ax0_ax1_ax2_ax3_fused_2: int32, 0, 2) {
                      for (ax0_ax1_ax2_ax3_fused_3: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_ax3_fused_4: int32, 0, 8) "vectorized" {
                          block([196, 36, 16, 16], "A_shared") as [v0, v1, v2, v3] {
                            bind(v0, ((ii_0*4) + floordiv((((((ax0_ax1_ax2_ax3_fused_0*1024) + (ax0_ax1_ax2_ax3_fused_1*512)) + (ax0_ax1_ax2_ax3_fused_2*256)) + (ax0_ax1_ax2_ax3_fused_3*8)) + ax0_ax1_ax2_ax3_fused_4), 512)))
                            bind(v1, ((kk_0*2) + floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0*1024) + (ax0_ax1_ax2_ax3_fused_1*512)) + (ax0_ax1_ax2_ax3_fused_2*256)) + (ax0_ax1_ax2_ax3_fused_3*8)) + ax0_ax1_ax2_ax3_fused_4), 512), 256)))
                            bind(v2, floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0*1024) + (ax0_ax1_ax2_ax3_fused_1*512)) + (ax0_ax1_ax2_ax3_fused_2*256)) + (ax0_ax1_ax2_ax3_fused_3*8)) + ax0_ax1_ax2_ax3_fused_4), 256), 16))
                            bind(v3, floormod((((((ax0_ax1_ax2_ax3_fused_0*1024) + (ax0_ax1_ax2_ax3_fused_1*512)) + (ax0_ax1_ax2_ax3_fused_2*256)) + (ax0_ax1_ax2_ax3_fused_3*8)) + ax0_ax1_ax2_ax3_fused_4), 16))
                            tir.reads([A[v0, v1, ((floormod(v2, 8)*2) + floordiv(v3, 8)), ((floordiv(v2, 8)*8) + floormod(v3, 8))]])
                            tir.writes([A_shared[v0, v1, v2, v3]])
                            A_shared[v0, v1, v2, v3] = A[v0, v1, ((floormod(v2, 8)*2) + floordiv(v3, 8)), ((floordiv(v2, 8)*8) + floormod(v3, 8))]
                        }
                      }
                    }
                  }
                }
                for (ax0_ax1_ax2_ax3_fused_0_1: int32, 0, 2) "thread_binding" {
                  for (ax0_ax1_ax2_ax3_fused_1_1: int32, 0, 2) "thread_binding" {
                    for (ax0_ax1_ax2_ax3_fused_2_1: int32, 0, 2) {
                      for (ax0_ax1_ax2_ax3_fused_3_1: int32, 0, 32) "thread_binding" {
                        for (ax0_ax1_ax2_ax3_fused_4_1: int32, 0, 8) "vectorized" {
                          block([36, 4, 16, 16], "B_shared") as [v0_1, v1_1, v2_1, v3_1] {
                            bind(v0_1, ((kk_0*2) + floordiv((((((ax0_ax1_ax2_ax3_fused_0_1*1024) + (ax0_ax1_ax2_ax3_fused_1_1*512)) + (ax0_ax1_ax2_ax3_fused_2_1*256)) + (ax0_ax1_ax2_ax3_fused_3_1*8)) + ax0_ax1_ax2_ax3_fused_4_1), 1024)))
                            bind(v1_1, floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0_1*1024) + (ax0_ax1_ax2_ax3_fused_1_1*512)) + (ax0_ax1_ax2_ax3_fused_2_1*256)) + (ax0_ax1_ax2_ax3_fused_3_1*8)) + ax0_ax1_ax2_ax3_fused_4_1), 1024), 256))
                            bind(v2_1, floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0_1*1024) + (ax0_ax1_ax2_ax3_fused_1_1*512)) + (ax0_ax1_ax2_ax3_fused_2_1*256)) + (ax0_ax1_ax2_ax3_fused_3_1*8)) + ax0_ax1_ax2_ax3_fused_4_1), 256), 16))
                            bind(v3_1, floormod((((((ax0_ax1_ax2_ax3_fused_0_1*1024) + (ax0_ax1_ax2_ax3_fused_1_1*512)) + (ax0_ax1_ax2_ax3_fused_2_1*256)) + (ax0_ax1_ax2_ax3_fused_3_1*8)) + ax0_ax1_ax2_ax3_fused_4_1), 16))
                            tir.reads([B[v0_1, v1_1, ((floormod(v2_1, 8)*2) + floordiv(v3_1, 8)), ((floordiv(v2_1, 8)*8) + floormod(v3_1, 8))]])
                            tir.writes([B_shared[v0_1, v1_1, v2_1, v3_1]])
                            B_shared[v0_1, v1_1, v2_1, v3_1] = B[v0_1, v1_1, ((floormod(v2_1, 8)*2) + floordiv(v3_1, 8)), ((floordiv(v2_1, 8)*8) + floormod(v3_1, 8))]
                        }
                      }
                    }
                  }
                }
                for (kk_1: int32, 0, 2) {
                  for (ax0: int32, 0, 2) {
                    block([196, 36, 1, 1], "A_shared_warp_o") as [v0_2, v1_2, v2_o, v3_o] {
                      bind(v0_2, (((ii_0*4) + (ii_1*2)) + ax0))
                      bind(v1_2, ((kk_0*2) + kk_1))
                      bind(v2_o, 0)
                      bind(v3_o, 0)
                      tir.reads([A_shared[v0_2, v1_2, 0:16, 0:16]])
                      tir.writes([A_shared_warp[v0_2, v1_2, 0:32, 0:8]])
                      warp = match_buffer(A_shared_warp[v0_2, v1_2, 0:32, 0:8])
                      shared = match_buffer(A_shared[v0_2, v1_2, 0:16, 0:16])
                      attr [IterVar(tx_1: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
                      @tir.ptx_ldmatrix(False, 4, ".b16", warp_1: Pointer(warp float16), (elem_offset_1: int32 + (8*tx_1)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_1: Pointer(shared float16), elem_offset_2: int32, (shared_s0: int32*16), 1, dtype=handle), (8*tx_1), dtype=float16)
                  }
                  for (ax0_1: int32, 0, 2) {
                    block([36, 4, 1, 1], "B_shared_warp_o") as [v0_3, v1_3, v2_o_1, v3_o_1] {
                      bind(v0_3, ((kk_0*2) + kk_1))
                      bind(v1_3, ((jj_1*2) + ax0_1))
                      bind(v2_o_1, 0)
                      bind(v3_o_1, 0)
                      tir.reads([B_shared[v0_3, v1_3, 0:16, 0:16]])
                      tir.writes([B_shared_warp[v0_3, v1_3, 0:32, 0:8]])
                      warp_2 = match_buffer(B_shared_warp[v0_3, v1_3, 0:32, 0:8])
                      shared_2 = match_buffer(B_shared[v0_3, v1_3, 0:16, 0:16])
                      attr [IterVar(tx_2: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
                      @tir.ptx_ldmatrix(True, 4, ".b16", warp_3: Pointer(warp float16), (elem_offset_3: int32 + (8*tx_2)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_3: Pointer(shared float16), elem_offset_4: int32, (shared_s0_1: int32*16), 1, dtype=handle), (8*tx_2), dtype=float16)
                  }
                  for (ii_2: int32, 0, 2) {
                    for (jj_2: int32, 0, 2) {
                      block([1, 196, 4, tir.reduce_axis(0, 36), 1, 1, tir.reduce_axis(0, 1)], "B_update_o") as [vsk_1, vii_1, vjj_1, vkk, vi_o_1, vj_o_1, vk_o] {
                        bind(vsk_1, sk)
                        bind(vii_1, (((ii_0*4) + (ii_1*2)) + ii_2))
                        bind(vjj_1, (((jj_0*4) + (jj_1*2)) + jj_2))
                        bind(vkk, ((kk_0*2) + kk_1))
                        bind(vi_o_1, 0)
                        bind(vj_o_1, 0)
                        bind(vk_o, 0)
                        tir.reads([C_warp[vsk_1, vii_1, vjj_1, 0:32, 0:8], A_shared_warp[vii_1, vkk, 0:32, 0:8], B_shared_warp[vkk, vjj_1, 0:32, 0:8]])
                        tir.writes([C_warp[vsk_1, vii_1, vjj_1, 0:32, 0:8]])
                        A_2 = match_buffer(A_shared_warp[vii_1, vkk, 0:32, 0:8])
                        B_2 = match_buffer(B_shared_warp[vkk, vjj_1, 0:32, 0:8])
                        C_2 = match_buffer(C_warp[vsk_1, vii_1, vjj_1, 0:32, 0:8])
                        attr [IterVar(tx_3: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32 {
                          @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_3: Pointer(warp float16), (elem_offset_5: int32 + (tx_3*8)), B_3: Pointer(warp float16), (elem_offset_6: int32 + (tx_3*8)), C_3: Pointer(warp float16), (elem_offset_7: int32 + (tx_3*8)), False, dtype=float16)
                          @tir.ptx_mma("m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_3, (elem_offset_5 + (tx_3*8)), B_3, ((elem_offset_6 + (tx_3*8)) + floordiv(8, 2)), C_3, ((elem_offset_7 + (tx_3*8)) + floordiv(8, 2)), False, dtype=float16)
                        }
                    }
                  }
                }
              }
              for (ax0_2: int32, 0, 2) {
                for (ax1: int32, 0, 2) {
                  block([1, 196, 4, 1, 1], "C_warp_o") as [v0_4, v1_4, v2_2, v3_o_2, v4_o] {
                    bind(v0_4, 0)
                    bind(v1_4, (((ii_0*4) + (ii_1*2)) + ax0_2))
                    bind(v2_2, ((jj_1*2) + ax1))
                    bind(v3_o_2, 0)
                    bind(v4_o, 0)
                    tir.reads([C_warp[v0_4, v1_4, v2_2, 0:32, 0:8]])
                    tir.writes([C[v0_4, v1_4, v2_2, 0:16, 0:16]])
                    C_warp_3 = match_buffer(C_warp[v0_4, v1_4, v2_2, 0:32, 0:8])
                    C_4 = match_buffer(C[v0_4, v1_4, v2_2, 0:16, 0:16])
                    attr [IterVar(tx_4: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
                    @tir.mma_store(16, 16, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_5: Pointer(global float16), elem_offset_8: int32, (C_s0: int32*16), 2, dtype=handle), C_warp_4: Pointer(warp float16), elem_offset_9: int32, C_s0, dtype=float16)
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