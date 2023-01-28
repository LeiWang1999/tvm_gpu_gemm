#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [1024, 1024, 16, 16], []),
             B: Buffer(B_1: Pointer(global float16), float16, [1024, 1024, 16, 16], []),
             C: Buffer(C_1: Pointer(global float16), float16, [1024, 1024, 16, 16], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared.dyn = alloc_buffer(float16[1024, 1024, 16, 16])
    A_shared.dyn_warp = alloc_buffer(float16[1024, 1024, 32, 8])
    B_shared.dyn = alloc_buffer(float16[1024, 1024, 16, 16])
    B_shared.dyn_warp = alloc_buffer(float16[1024, 1024, 32, 8])
    C_warp = alloc_buffer(float16[1024, 1024, 32, 8])
    for (ii_0: int32, 0, 128) "thread_binding" {
      for (jj_0: int32, 0, 64) "thread_binding" {
        for (ii_1: int32, 0, 4) "thread_binding" {
          for (jj_1: int32, 0, 2) "thread_binding" {
            for (ii_2_init: int32, 0, 2) {
              for (jj_2_init: int32, 0, 8) {
                block([1024, 1024, 1, 1], "B_init_o") as [vii, vjj, vi_o, vj_o] {
                  bind(vii, (((ii_0*8) + (ii_1*2)) + ii_2_init))
                  bind(vjj, (((jj_0*16) + (jj_1*8)) + jj_2_init))
                  bind(vi_o, 0)
                  bind(vj_o, 0)
                  tir.reads([])
                  tir.writes([C_warp[vii, vjj, 0:32, 0:8]])
                  C_warp_1 = match_buffer(C_warp[vii, vjj, 0:32, 0:8])
                  attr [IterVar(tx: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
                  @tir.mma_fill(8, C_warp_2: Pointer(warp float16), elem_offset: int32, dtype=float16)
              }
            }
            for (kk_0: int32, 0, 512) {
              for (ax0_ax1_ax2_ax3_fused_0: int32, 0, 4) "thread_binding" {
                for (ax0_ax1_ax2_ax3_fused_1: int32, 0, 2) "thread_binding" {
                  for (ax0_ax1_ax2_ax3_fused_2: int32, 0, 2) {
                    for (ax0_ax1_ax2_ax3_fused_3: int32, 0, 32) "thread_binding" {
                      for (ax0_ax1_ax2_ax3_fused_4: int32, 0, 8) "vectorized" {
                        block([1024, 1024, 16, 16], "A_shared.dyn") as [v0, v1, v2, v3] {
                          bind(v0, ((ii_0*8) + floordiv((((((ax0_ax1_ax2_ax3_fused_0*1024) + (ax0_ax1_ax2_ax3_fused_1*512)) + (ax0_ax1_ax2_ax3_fused_2*256)) + (ax0_ax1_ax2_ax3_fused_3*8)) + ax0_ax1_ax2_ax3_fused_4), 512)))
                          bind(v1, ((kk_0*2) + floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0*1024) + (ax0_ax1_ax2_ax3_fused_1*512)) + (ax0_ax1_ax2_ax3_fused_2*256)) + (ax0_ax1_ax2_ax3_fused_3*8)) + ax0_ax1_ax2_ax3_fused_4), 512), 256)))
                          bind(v2, floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0*1024) + (ax0_ax1_ax2_ax3_fused_1*512)) + (ax0_ax1_ax2_ax3_fused_2*256)) + (ax0_ax1_ax2_ax3_fused_3*8)) + ax0_ax1_ax2_ax3_fused_4), 256), 16))
                          bind(v3, floormod((((((ax0_ax1_ax2_ax3_fused_0*1024) + (ax0_ax1_ax2_ax3_fused_1*512)) + (ax0_ax1_ax2_ax3_fused_2*256)) + (ax0_ax1_ax2_ax3_fused_3*8)) + ax0_ax1_ax2_ax3_fused_4), 16))
                          tir.reads([A[v0, v1, ((floormod(v2, 8)*2) + floordiv(v3, 8)), ((floordiv(v2, 8)*8) + floormod(v3, 8))]])
                          tir.writes([A_shared.dyn[v0, v1, v2, v3]])
                          A_shared.dyn[v0, v1, v2, v3] = A[v0, v1, ((floormod(v2, 8)*2) + floordiv(v3, 8)), ((floordiv(v2, 8)*8) + floormod(v3, 8))]
                      }
                    }
                  }
                }
              }
              for (ax0_ax1_ax2_ax3_fused_0_1: int32, 0, 4) "thread_binding" {
                for (ax0_ax1_ax2_ax3_fused_1_1: int32, 0, 2) "thread_binding" {
                  for (ax0_ax1_ax2_ax3_fused_2_1: int32, 0, 4) {
                    for (ax0_ax1_ax2_ax3_fused_3_1: int32, 0, 32) "thread_binding" {
                      for (ax0_ax1_ax2_ax3_fused_4_1: int32, 0, 8) "vectorized" {
                        block([1024, 1024, 16, 16], "B_shared.dyn") as [v0_1, v1_1, v2_1, v3_1] {
                          bind(v0_1, ((jj_0*16) + floordiv((((((ax0_ax1_ax2_ax3_fused_0_1*2048) + (ax0_ax1_ax2_ax3_fused_1_1*1024)) + (ax0_ax1_ax2_ax3_fused_2_1*256)) + (ax0_ax1_ax2_ax3_fused_3_1*8)) + ax0_ax1_ax2_ax3_fused_4_1), 512)))
                          bind(v1_1, ((kk_0*2) + floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0_1*2048) + (ax0_ax1_ax2_ax3_fused_1_1*1024)) + (ax0_ax1_ax2_ax3_fused_2_1*256)) + (ax0_ax1_ax2_ax3_fused_3_1*8)) + ax0_ax1_ax2_ax3_fused_4_1), 512), 256)))
                          bind(v2_1, floordiv(floormod((((((ax0_ax1_ax2_ax3_fused_0_1*2048) + (ax0_ax1_ax2_ax3_fused_1_1*1024)) + (ax0_ax1_ax2_ax3_fused_2_1*256)) + (ax0_ax1_ax2_ax3_fused_3_1*8)) + ax0_ax1_ax2_ax3_fused_4_1), 256), 16))
                          bind(v3_1, floormod((((((ax0_ax1_ax2_ax3_fused_0_1*2048) + (ax0_ax1_ax2_ax3_fused_1_1*1024)) + (ax0_ax1_ax2_ax3_fused_2_1*256)) + (ax0_ax1_ax2_ax3_fused_3_1*8)) + ax0_ax1_ax2_ax3_fused_4_1), 16))
                          tir.reads([B[v0_1, v1_1, ((floormod(v2_1, 8)*2) + floordiv(v3_1, 8)), ((floordiv(v2_1, 8)*8) + floormod(v3_1, 8))]])
                          tir.writes([B_shared.dyn[v0_1, v1_1, v2_1, v3_1]])
                          B_shared.dyn[v0_1, v1_1, v2_1, v3_1] = B[v0_1, v1_1, ((floormod(v2_1, 8)*2) + floordiv(v3_1, 8)), ((floordiv(v2_1, 8)*8) + floormod(v3_1, 8))]
                      }
                    }
                  }
                }
              }
              for (kk_1: int32, 0, 2) {
                for (ax0: int32, 0, 2) {
                  block([1024, 1024, 1, 1], "A_shared.dyn_warp_o") as [v0_2, v1_2, v2_o, v3_o] {
                    bind(v0_2, (((ii_0*8) + (ii_1*2)) + ax0))
                    bind(v1_2, ((kk_0*2) + kk_1))
                    bind(v2_o, 0)
                    bind(v3_o, 0)
                    tir.reads([A_shared.dyn[v0_2, v1_2, 0:16, 0:16]])
                    tir.writes([A_shared.dyn_warp[v0_2, v1_2, 0:32, 0:8]])
                    warp = match_buffer(A_shared.dyn_warp[v0_2, v1_2, 0:32, 0:8])
                    shared = match_buffer(A_shared.dyn[v0_2, v1_2, 0:16, 0:16])
                    attr [IterVar(tx_1: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
                    @tir.ptx_ldmatrix(False, 4, ".b16", warp_1: Pointer(warp float16), (elem_offset_1: int32 + (8*tx_1)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_1: Pointer(shared.dyn float16), elem_offset_2: int32, (shared_s0: int32*16), 1, dtype=handle), (8*tx_1), dtype=float16)
                }
                for (ax0_1: int32, 0, 8) {
                  for (ax1: int32, 0, 16) {
                    for (ax2: int32, 0, 16) {
                      block([1024, 1024, 16, 16], "B_shared.dyn_warp") as [v0_3, v1_3, v2_2, v3_2] {
                        bind(v0_3, (((jj_0*16) + (jj_1*8)) + ax0_1))
                        bind(v1_3, ((kk_0*2) + kk_1))
                        bind(v2_2, ax1)
                        bind(v3_2, ax2)
                        tir.reads([B_shared.dyn[v0_3, v1_3, v2_2, v3_2]])
                        tir.writes([B_shared.dyn_warp[v0_3, v1_3, ((v2_2*2) + floordiv(v3_2, 8)), floormod(v3_2, 8)]])
                        B_shared.dyn_warp[v0_3, v1_3, ((v2_2*2) + floordiv(v3_2, 8)), floormod(v3_2, 8)] = B_shared.dyn[v0_3, v1_3, v2_2, v3_2]
                    }
                  }
                }
                for (ii_2: int32, 0, 2) {
                  for (jj_2: int32, 0, 8) {
                    for (i: int32, 0, 16) {
                      for (j: int32, 0, 16) {
                        for (k: int32, 0, 16) {
                          block([1024, 1024, tir.reduce_axis(0, 1024), 16, 16, tir.reduce_axis(0, 16)], "B_update") as [vii_1, vjj_1, vkk, vi, vj, vk] {
                            bind(vii_1, (((ii_0*8) + (ii_1*2)) + ii_2))
                            bind(vjj_1, (((jj_0*16) + (jj_1*8)) + jj_2))
                            bind(vkk, ((kk_0*2) + kk_1))
                            bind(vi, i)
                            bind(vj, j)
                            bind(vk, k)
                            tir.reads([C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))], A_shared.dyn_warp[vii_1, vkk, ((vi*2) + floordiv(vk, 8)), floormod(vk, 8)], B_shared.dyn_warp[vjj_1, vkk, ((vj*2) + floordiv(vk, 8)), floormod(vk, 8)]])
                            tir.writes([C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))]])
                            C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))] = (C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))] + (A_shared.dyn_warp[vii_1, vkk, ((vi*2) + floordiv(vk, 8)), floormod(vk, 8)]*B_shared.dyn_warp[vjj_1, vkk, ((vj*2) + floordiv(vk, 8)), floormod(vk, 8)]))
                        }
                      }
                    }
                  }
                }
              }
            }
            for (ax0_2: int32, 0, 2) {
              for (ax1_1: int32, 0, 8) {
                for (ax2_1: int32, 0, 16) {
                  for (ax3: int32, 0, 16) {
                    block([1024, 1024, 16, 16], "C_warp") as [v0_4, v1_4, v2_3, v3_3] {
                      bind(v0_4, (((ii_0*8) + (ii_1*2)) + ax0_2))
                      bind(v1_4, (((jj_0*16) + (jj_1*8)) + ax1_1))
                      bind(v2_3, ax2_1)
                      bind(v3_3, ax3)
                      tir.reads([C_warp[v0_4, v1_4, ((floormod(v2_3, 8)*4) + floordiv(floormod(v3_3, 8), 2)), (((floordiv(v3_3, 8)*4) + (floordiv(v2_3, 8)*2)) + floormod(v3_3, 2))]])
                      tir.writes([C[v0_4, v1_4, v2_3, v3_3]])
                      C[v0_4, v1_4, v2_3, v3_3] = C_warp[v0_4, v1_4, ((floormod(v2_3, 8)*4) + floordiv(floormod(v3_3, 8), 2)), (((floordiv(v3_3, 8)*4) + (floordiv(v2_3, 8)*2)) + floormod(v3_3, 2))]
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