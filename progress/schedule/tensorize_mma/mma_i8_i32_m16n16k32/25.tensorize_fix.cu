#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [1, 1, 16, 32], []),
             B: Buffer(B_1: Pointer(global int8), int8, [1, 1, 16, 32], []),
             C: Buffer(C_1: Pointer(global int32), int32, [1, 1, 16, 16], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(int8[1, 1, 16, 32])
    A_shared_warp = alloc_buffer(int8[1, 1, 32, 16])
    B_shared = alloc_buffer(int8[1, 1, 16, 32])
    B_shared_warp = alloc_buffer(int8[1, 1, 32, 16])
    B_shared_warp_warp = alloc_buffer(int8[1, 1, 32, 16])
    C_warp = alloc_buffer(int32[1, 1, 32, 8])
    for (ii: int32, 0, 1) "thread_binding" {
      for (jj: int32, 0, 1) "thread_binding" {
        for (ax0_ax1_fused_0: int32, 0, 1) "thread_binding" {
          for (ax0_ax1_fused_1: int32, 0, 1) "thread_binding" {
            for (ax0_ax1_fused_2: int32, 0, 1) {
              for (ax0_ax1_fused_3: int32, 0, 32) "thread_binding" {
                for (ax0_ax1_fused_4: int32, 0, 16) "vectorized" {
                  block([1, 1, 16, 32], "A_shared") as [v0, v1, v2, v3] {
                    bind(v0, 0)
                    bind(v1, 0)
                    bind(v2, floordiv((((((ax0_ax1_fused_0*512) + (ax0_ax1_fused_1*512)) + (ax0_ax1_fused_2*512)) + (ax0_ax1_fused_3*16)) + ax0_ax1_fused_4), 32))
                    bind(v3, floormod((((((ax0_ax1_fused_0*512) + (ax0_ax1_fused_1*512)) + (ax0_ax1_fused_2*512)) + (ax0_ax1_fused_3*16)) + ax0_ax1_fused_4), 32))
                    tir.reads([A[v0, v1, ((floormod(v2, 8)*2) + floordiv(v3, 16)), ((floordiv(v2, 8)*16) + floormod(v3, 16))]])
                    tir.writes([A_shared[v0, v1, v2, v3]])
                    A_shared[v0, v1, v2, v3] = A[v0, v1, ((floormod(v2, 8)*2) + floordiv(v3, 16)), ((floordiv(v2, 8)*16) + floormod(v3, 16))]
                }
              }
            }
          }
        }
        block([1, 1, 1, 1], "A_shared_warp_o") as [v0_1, v1_1, v2_o, v3_o] {
          bind(v0_1, 0)
          bind(v1_1, 0)
          bind(v2_o, 0)
          bind(v3_o, 0)
          tir.reads([A_shared[v0_1, v1_1, 0:16, 0:32]])
          tir.writes([A_shared_warp[v0_1, v1_1, 0:32, 0:16]])
          warp = match_buffer(A_shared_warp[v0_1, v1_1, 0:32, 0:16])
          shared = match_buffer(A_shared[v0_1, v1_1, 0:16, 0:32])
          attr [IterVar(tx: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
          @tir.ptx_ldmatrix(False, 4, ".b16", warp_1: Pointer(warp int8), (elem_offset: int32 + (16*tx)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), shared_1: Pointer(shared int8), elem_offset_1: int32, (shared_s0: int32*16), 1, dtype=handle), (16*tx), dtype=int8)
        for (ax0_ax1_fused_0_1: int32, 0, 1) "thread_binding" {
          for (ax0_ax1_fused_1_1: int32, 0, 1) "thread_binding" {
            for (ax0_ax1_fused_2_1: int32, 0, 1) {
              for (ax0_ax1_fused_3_1: int32, 0, 32) "thread_binding" {
                for (ax0_ax1_fused_4_1: int32, 0, 16) "vectorized" {
                  block([1, 1, 16, 32], "B_shared") as [v0_2, v1_2, v2_1, v3_1] {
                    bind(v0_2, 0)
                    bind(v1_2, 0)
                    bind(v2_1, floordiv((((((ax0_ax1_fused_0_1*512) + (ax0_ax1_fused_1_1*512)) + (ax0_ax1_fused_2_1*512)) + (ax0_ax1_fused_3_1*16)) + ax0_ax1_fused_4_1), 32))
                    bind(v3_1, floormod((((((ax0_ax1_fused_0_1*512) + (ax0_ax1_fused_1_1*512)) + (ax0_ax1_fused_2_1*512)) + (ax0_ax1_fused_3_1*16)) + ax0_ax1_fused_4_1), 32))
                    tir.reads([B[v0_2, v1_2, ((floormod(v2_1, 8)*2) + floordiv(v3_1, 16)), ((floordiv(v2_1, 8)*16) + floormod(v3_1, 16))]])
                    tir.writes([B_shared[v0_2, v1_2, v2_1, v3_1]])
                    B_shared[v0_2, v1_2, v2_1, v3_1] = B[v0_2, v1_2, ((floormod(v2_1, 8)*2) + floordiv(v3_1, 16)), ((floordiv(v2_1, 8)*16) + floormod(v3_1, 16))]
                }
              }
            }
          }
        }
        block([1, 1, 1, 1], "B_shared_warp_o") as [v0_3, v1_3, v2_o_1, v3_o_1] {
          bind(v0_3, 0)
          bind(v1_3, 0)
          bind(v2_o_1, 0)
          bind(v3_o_1, 0)
          tir.reads([B_shared[v0_3, v1_3, 0:16, 0:32]])
          tir.writes([B_shared_warp[v0_3, v1_3, 0:32, 0:16]])
          warp_2 = match_buffer(B_shared_warp[v0_3, v1_3, 0:32, 0:16])
          shared_2 = match_buffer(B_shared[v0_3, v1_3, 0:16, 0:32])
          attr [IterVar(tx_1: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
          @tir.ptx_ldmatrix(False, 4, ".b16", warp_3: Pointer(warp int8), (elem_offset_2: int32 + (16*tx_1)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), shared_3: Pointer(shared int8), elem_offset_3: int32, (shared_s0_1: int32*16), 1, dtype=handle), (16*tx_1), dtype=int8)
        block([1, 1, 1, 1], "B_shared_warp_warp_o") as [v0_4, v1_4, v2_o_2, v3_o_2] {
          bind(v0_4, 0)
          bind(v1_4, 0)
          bind(v2_o_2, 0)
          bind(v3_o_2, 0)
          tir.reads([B_shared_warp[v0_4, v1_4, 0:32, 0:16]])
          tir.writes([B_shared_warp_warp[v0_4, v1_4, 0:32, 0:16]])
          B_warp = match_buffer(B_shared_warp[v0_4, v1_4, 0:32, 0:16])
          B_warp_permutated = match_buffer(B_shared_warp_warp[v0_4, v1_4, 0:32, 0:16])
          attr [IterVar(tx_2: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
          for (i0: int32, 0, 32) {
            block([32], "B_warp_warp") as [v0_5] {
              bind(v0_5, i0)
              tir.reads([B_warp[v0_5, 0:16]])
              tir.writes([B_warp_permutated[v0_5, 0:16]])
               {
                B_warp_permutated[v0_5, 0] = B_warp[v0_5, 8]
                B_warp_permutated[v0_5, 1] = B_warp[v0_5, 9]
                B_warp_permutated[v0_5, 2] = B_warp[v0_5, 10]
                B_warp_permutated[v0_5, 3] = B_warp[v0_5, 11]
                B_warp_permutated[v0_5, 4] = B_warp[v0_5, 12]
              }
          }
        block([1, 1, 1, 1], "B_init_o") as [vii, vjj, vi_o, vj_o] {
          bind(vii, 0)
          bind(vjj, 0)
          bind(vi_o, 0)
          bind(vj_o, 0)
          tir.reads([])
          tir.writes([C_warp[vii, vjj, 0:32, 0:8]])
          C_warp_1 = match_buffer(C_warp[vii, vjj, 0:32, 0:8])
          attr [IterVar(tx_3: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
          @tir.mma_fill(8, C_warp_2: Pointer(warp int32), elem_offset_4: int32, dtype=int32)
        for (kk: int32, 0, 1) {
          for (i: int32, 0, 16) {
            for (j: int32, 0, 16) {
              for (k: int32, 0, 32) {
                block([1, 1, tir.reduce_axis(0, 1), 16, 16, tir.reduce_axis(0, 32)], "B_update") as [vii_1, vjj_1, vkk, vi, vj, vk] {
                  bind(vii_1, ii)
                  bind(vjj_1, jj)
                  bind(vkk, kk)
                  bind(vi, i)
                  bind(vj, j)
                  bind(vk, k)
                  tir.reads([C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))], A_shared_warp[vii_1, vkk, ((vi*2) + floordiv(vk, 16)), floormod(vk, 16)], B_shared_warp_warp[vjj_1, vkk, ((vj*2) + floordiv(vk, 16)), floormod(vk, 16)]])
                  tir.writes([C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))]])
                  C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))] = (C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))] + (cast(int32, A_shared_warp[vii_1, vkk, ((vi*2) + floordiv(vk, 16)), floormod(vk, 16)])*cast(int32, B_shared_warp_warp[vjj_1, vkk, ((vj*2) + floordiv(vk, 16)), floormod(vk, 16)])))
              }
            }
          }
        }
        for (ax0: int32, 0, 16) {
          for (ax1: int32, 0, 16) {
            block([1, 1, 16, 16], "C_warp") as [v0_6, v1_5, v2_2, v3_2] {
              bind(v0_6, 0)
              bind(v1_5, 0)
              bind(v2_2, ax0)
              bind(v3_2, ax1)
              tir.reads([C_warp[v0_6, v1_5, ((floormod(v2_2, 8)*4) + floordiv(floormod(v3_2, 8), 2)), (((floordiv(v3_2, 8)*4) + (floordiv(v2_2, 8)*2)) + floormod(v3_2, 2))]])
              tir.writes([C[v0_6, v1_5, v2_2, v3_2]])
              C[v0_6, v1_5, v2_2, v3_2] = C_warp[v0_6, v1_5, ((floormod(v2_2, 8)*4) + floordiv(floormod(v3_2, 8), 2)), (((floordiv(v3_2, 8)*4) + (floordiv(v2_2, 8)*2)) + floormod(v3_2, 2))]
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