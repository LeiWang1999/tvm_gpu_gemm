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
    A_shared = alloc_buffer(float16[16384, 16384])
    B_shared = alloc_buffer(float16[16384, 16384])
    A_shared_warp = alloc_buffer(float16[1024, 1024, 32, 8])
    B_shared_warp = alloc_buffer(float16[1024, 1024, 32, 8])
    C_warp = alloc_buffer(float16[1024, 1024, 32, 8])
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
                      tir.writes([C_warp[floordiv(vi, 16), floordiv(vj, 16), ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(floormod(vj, 16), 8)*4) + (floordiv(floormod(vi, 16), 8)*2)) + floormod(vj, 2))]])
                      C_warp[floordiv(vi, 16), floordiv(vj, 16), ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(floormod(vj, 16), 8)*4) + (floordiv(floormod(vi, 16), 8)*2)) + floormod(vj, 2))] = 0f16
                  }
                }
              }
            }
            for (k_0: int32, 0, 512) {
              for (ax0_ax1_fused_0: int32, 0, 2) {
                for (ax0_ax1_fused_1: int32, 0, 2) "thread_binding" {
                  for (ax0_ax1_fused_2: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_fused_3: int32, 0, 32) "thread_binding" {
                      for (ax0_ax1_fused_4: int32, 0, 8) "vectorized" {
                        block([16384, 16384], "A_shared") as [v0, v1] {
                          bind(v0, ((i_0*128) + floordiv((((((ax0_ax1_fused_0*2048) + (ax0_ax1_fused_1*1024)) + (ax0_ax1_fused_2*256)) + (ax0_ax1_fused_3*8)) + ax0_ax1_fused_4), 32)))
                          bind(v1, ((k_0*32) + floormod((((((ax0_ax1_fused_0*2048) + (ax0_ax1_fused_1*1024)) + (ax0_ax1_fused_2*256)) + (ax0_ax1_fused_3*8)) + ax0_ax1_fused_4), 32)))
                          tir.reads([A[v0, v1]])
                          tir.writes([A_shared[v0, v1]])
                          tir.attrs({"buffer_dim_align": [[0, 0, 32, 8]]})
                          A_shared[v0, v1] = A[v0, v1]
                      }
                    }
                  }
                }
              }
              for (ax0_ax1_fused_0_1: int32, 0, 4) {
                for (ax0_ax1_fused_1_1: int32, 0, 2) "thread_binding" {
                  for (ax0_ax1_fused_2_1: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_fused_3_1: int32, 0, 32) "thread_binding" {
                      for (ax0_ax1_fused_4_1: int32, 0, 8) "vectorized" {
                        block([16384, 16384], "B_shared") as [v0_1, v1_1] {
                          bind(v0_1, ((j_0*256) + floordiv((((((ax0_ax1_fused_0_1*2048) + (ax0_ax1_fused_1_1*1024)) + (ax0_ax1_fused_2_1*256)) + (ax0_ax1_fused_3_1*8)) + ax0_ax1_fused_4_1), 32)))
                          bind(v1_1, ((k_0*32) + floormod((((((ax0_ax1_fused_0_1*2048) + (ax0_ax1_fused_1_1*1024)) + (ax0_ax1_fused_2_1*256)) + (ax0_ax1_fused_3_1*8)) + ax0_ax1_fused_4_1), 32)))
                          tir.reads([B[v0_1, v1_1]])
                          tir.writes([B_shared[v0_1, v1_1]])
                          tir.attrs({"buffer_dim_align": [[0, 0, 32, 8]]})
                          B_shared[v0_1, v1_1] = B[v0_1, v1_1]
                      }
                    }
                  }
                }
              }
              for (i_1_1_0: int32, 0, 4) {
                for (j_1_1_0: int32, 0, 4) {
                  for (k_1_0: int32, 0, 2) {
                    block([1024, 1024], "A_shared_warp_o") as [v0_o, v1_o] {
                      bind(v0_o, (((i_0*8) + (i_1_0*4)) + i_1_1_0))
                      bind(v1_o, ((k_0*2) + k_1_0))
                      tir.reads([A_shared[(v0_o*16):((v0_o*16) + 16), (v1_o*16):((v1_o*16) + 16)]])
                      tir.writes([A_shared_warp[v0_o, v1_o, 0:32, 0:8]])
                      warp = match_buffer(A_shared_warp[v0_o, v1_o, 0:32, 0:8])
                      shared = match_buffer(A_shared[(v0_o*16):((v0_o*16) + 16), (v1_o*16):((v1_o*16) + 16)])
                      attr [IterVar(tx: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
                      @tir.ptx_ldmatrix(False, 4, ".b16", warp_1: Pointer(warp float16), (elem_offset: int32 + (8*tx)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_1: Pointer(shared float16), elem_offset_1: int32, (shared_s0: int32*16), 1, dtype=handle), ((shared_s0*floormod(tx, 16)) + (8*floordiv(tx, 16))), dtype=float16)
                    block([1024, 1024], "B_shared_warp_o") as [v0_o_1, v1_o_1] {
                      bind(v0_o_1, (((j_0*16) + (j_1_0*4)) + j_1_1_0))
                      bind(v1_o_1, ((k_0*2) + k_1_0))
                      tir.reads([B_shared[(v0_o_1*16):((v0_o_1*16) + 16), (v1_o_1*16):((v1_o_1*16) + 16)]])
                      tir.writes([B_shared_warp[v0_o_1, v1_o_1, 0:32, 0:8]])
                      warp_2 = match_buffer(B_shared_warp[v0_o_1, v1_o_1, 0:32, 0:8])
                      shared_2 = match_buffer(B_shared[(v0_o_1*16):((v0_o_1*16) + 16), (v1_o_1*16):((v1_o_1*16) + 16)])
                      attr [IterVar(tx_1: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
                      @tir.ptx_ldmatrix(False, 4, ".b16", warp_3: Pointer(warp float16), (elem_offset_2: int32 + (8*tx_1)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), shared_3: Pointer(shared float16), elem_offset_3: int32, (shared_s0_1: int32*16), 1, dtype=handle), ((((shared_s0_1*8)*floordiv(tx_1, 16)) + (shared_s0_1*floormod(tx_1, 8))) + (8*floordiv(floormod(tx_1, 16), 8))), dtype=float16)
                    for (i_1_1_1: int32, 0, 16) {
                      for (j_1_1_1: int32, 0, 16) {
                        for (k_1_1: int32, 0, 16) {
                          block([16384, 16384, tir.reduce_axis(0, 16384)], "B_update") as [vi_1, vj_1, vk] {
                            bind(vi_1, ((((i_0*128) + (i_1_0*64)) + (i_1_1_0*16)) + i_1_1_1))
                            bind(vj_1, ((((j_0*256) + (j_1_0*64)) + (j_1_1_0*16)) + j_1_1_1))
                            bind(vk, (((k_0*32) + (k_1_0*16)) + k_1_1))
                            tir.reads([C_warp[floordiv(vi_1, 16), floordiv(vj_1, 16), ((floormod(vi_1, 8)*4) + floordiv(floormod(vj_1, 8), 2)), (((floordiv(floormod(vj_1, 16), 8)*4) + (floordiv(floormod(vi_1, 16), 8)*2)) + floormod(vj_1, 2))], A_shared_warp[floordiv(vi_1, 16), floordiv(vk, 16), ((floormod(vi_1, 8)*4) + floordiv(floormod(vk, 8), 2)), (((floordiv(floormod(vk, 16), 8)*4) + (floordiv(floormod(vi_1, 16), 8)*2)) + floormod(vk, 2))], B_shared_warp[floordiv(vj_1, 16), floordiv(vk, 16), ((floormod(vj_1, 8)*4) + floordiv(floormod(vk, 8), 2)), (((floordiv(floormod(vk, 16), 8)*4) + (floordiv(floormod(vj_1, 16), 8)*2)) + floormod(vk, 2))]])
                            tir.writes([C_warp[floordiv(vi_1, 16), floordiv(vj_1, 16), ((floormod(vi_1, 8)*4) + floordiv(floormod(vj_1, 8), 2)), (((floordiv(floormod(vj_1, 16), 8)*4) + (floordiv(floormod(vi_1, 16), 8)*2)) + floormod(vj_1, 2))]])
                            C_warp[floordiv(vi_1, 16), floordiv(vj_1, 16), ((floormod(vi_1, 8)*4) + floordiv(floormod(vj_1, 8), 2)), (((floordiv(floormod(vj_1, 16), 8)*4) + (floordiv(floormod(vi_1, 16), 8)*2)) + floormod(vj_1, 2))] = (C_warp[floordiv(vi_1, 16), floordiv(vj_1, 16), ((floormod(vi_1, 8)*4) + floordiv(floormod(vj_1, 8), 2)), (((floordiv(floormod(vj_1, 16), 8)*4) + (floordiv(floormod(vi_1, 16), 8)*2)) + floormod(vj_1, 2))] + (A_shared_warp[floordiv(vi_1, 16), floordiv(vk, 16), ((floormod(vi_1, 8)*4) + floordiv(floormod(vk, 8), 2)), (((floordiv(floormod(vk, 16), 8)*4) + (floordiv(floormod(vi_1, 16), 8)*2)) + floormod(vk, 2))]*B_shared_warp[floordiv(vj_1, 16), floordiv(vk, 16), ((floormod(vj_1, 8)*4) + floordiv(floormod(vk, 8), 2)), (((floordiv(floormod(vk, 16), 8)*4) + (floordiv(floormod(vj_1, 16), 8)*2)) + floormod(vk, 2))]))
                        }
                      }
                    }
                  }
                }
              }
            }
            for (ax0_0: int32, 0, 4) {
              for (ax1_0: int32, 0, 4) {
                for (ax0_1: int32, 0, 16) {
                  for (ax1_1: int32, 0, 16) {
                    block([16384, 16384], "C_warp") as [v0_2, v1_2] {
                      bind(v0_2, ((((i_0*128) + (i_1_0*64)) + (ax0_0*16)) + ax0_1))
                      bind(v1_2, ((((j_0*256) + (j_1_0*64)) + (ax1_0*16)) + ax1_1))
                      tir.reads([C_warp[floordiv(v0_2, 16), floordiv(v1_2, 16), ((floormod(v0_2, 8)*4) + floordiv(floormod(v1_2, 8), 2)), (((floordiv(floormod(v1_2, 16), 8)*4) + (floordiv(floormod(v0_2, 16), 8)*2)) + floormod(v1_2, 2))]])
                      tir.writes([C[v0_2, v1_2]])
                      C[v0_2, v1_2] = C_warp[floordiv(v0_2, 16), floordiv(v1_2, 16), ((floormod(v0_2, 8)*4) + floordiv(floormod(v1_2, 8), 2)), (((floordiv(floormod(v1_2, 16), 8)*4) + (floordiv(floormod(v0_2, 16), 8)*2)) + floormod(v1_2, 2))]
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