#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [1, 1, 16, 16], []),
             B: Buffer(B_1: Pointer(global float16), float16, [1, 1, 16, 16], []),
             C: Buffer(C_1: Pointer(global float16), float16, [1, 1, 16, 16], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(float16[1, 1, 16, 16])
    A_shared_warp = alloc_buffer(float16[1, 1, 32, 8])
    B_shared = alloc_buffer(float16[1, 1, 16, 16])
    B_shared_warp = alloc_buffer(float16[1, 1, 32, 8])
    C_warp = alloc_buffer(float16[1, 1, 32, 8])
    for (ii: int32, 0, 1) "thread_binding" {
      for (jj: int32, 0, 1) "thread_binding" {
        for (ax0_ax1_fused_0: int32, 0, 1) "thread_binding" {
          for (ax0_ax1_fused_1: int32, 0, 1) "thread_binding" {
            for (ax0_ax1_fused_2: int32, 0, 1) {
              for (ax0_ax1_fused_3: int32, 0, 32) "thread_binding" {
                for (ax0_ax1_fused_4: int32, 0, 8) "vectorized" {
                  block([1, 1, 16, 16], "A_shared") as [v0, v1, v2, v3] {
                    bind(v0, 0)
                    bind(v1, 0)
                    bind(v2, floordiv((((((ax0_ax1_fused_0*256) + (ax0_ax1_fused_1*256)) + (ax0_ax1_fused_2*256)) + (ax0_ax1_fused_3*8)) + ax0_ax1_fused_4), 16))
                    bind(v3, floormod((((((ax0_ax1_fused_0*256) + (ax0_ax1_fused_1*256)) + (ax0_ax1_fused_2*256)) + (ax0_ax1_fused_3*8)) + ax0_ax1_fused_4), 16))
                    tir.reads([A[v0, v1, ((floormod(v2, 8)*2) + floordiv(v3, 8)), ((floordiv(v2, 8)*8) + floormod(v3, 8))]])
                    tir.writes([A_shared[v0, v1, v2, v3]])
                    A_shared[v0, v1, v2, v3] = A[v0, v1, ((floormod(v2, 8)*2) + floordiv(v3, 8)), ((floordiv(v2, 8)*8) + floormod(v3, 8))]
                }
              }
            }
          }
        }
        for (ax0: int32, 0, 16) {
          for (ax1: int32, 0, 16) {
            block([1, 1, 16, 16], "A_shared_warp") as [v0_1, v1_1, v2_1, v3_1] {
              bind(v0_1, 0)
              bind(v1_1, 0)
              bind(v2_1, ax0)
              bind(v3_1, ax1)
              tir.reads([A_shared[v0_1, v1_1, v2_1, v3_1]])
              tir.writes([A_shared_warp[v0_1, v1_1, ((v2_1*2) + floordiv(v3_1, 8)), floormod(v3_1, 8)]])
              A_shared_warp[v0_1, v1_1, ((v2_1*2) + floordiv(v3_1, 8)), floormod(v3_1, 8)] = A_shared[v0_1, v1_1, v2_1, v3_1]
          }
        }
        for (ax0_ax1_fused_0_1: int32, 0, 1) "thread_binding" {
          for (ax0_ax1_fused_1_1: int32, 0, 1) "thread_binding" {
            for (ax0_ax1_fused_2_1: int32, 0, 1) {
              for (ax0_ax1_fused_3_1: int32, 0, 32) "thread_binding" {
                for (ax0_ax1_fused_4_1: int32, 0, 8) "vectorized" {
                  block([1, 1, 16, 16], "B_shared") as [v0_2, v1_2, v2_2, v3_2] {
                    bind(v0_2, 0)
                    bind(v1_2, 0)
                    bind(v2_2, floordiv((((((ax0_ax1_fused_0_1*256) + (ax0_ax1_fused_1_1*256)) + (ax0_ax1_fused_2_1*256)) + (ax0_ax1_fused_3_1*8)) + ax0_ax1_fused_4_1), 16))
                    bind(v3_2, floormod((((((ax0_ax1_fused_0_1*256) + (ax0_ax1_fused_1_1*256)) + (ax0_ax1_fused_2_1*256)) + (ax0_ax1_fused_3_1*8)) + ax0_ax1_fused_4_1), 16))
                    tir.reads([B[v0_2, v1_2, ((floormod(v2_2, 8)*2) + floordiv(v3_2, 8)), ((floordiv(v2_2, 8)*8) + floormod(v3_2, 8))]])
                    tir.writes([B_shared[v0_2, v1_2, v2_2, v3_2]])
                    B_shared[v0_2, v1_2, v2_2, v3_2] = B[v0_2, v1_2, ((floormod(v2_2, 8)*2) + floordiv(v3_2, 8)), ((floordiv(v2_2, 8)*8) + floormod(v3_2, 8))]
                }
              }
            }
          }
        }
        for (ax0_1: int32, 0, 16) {
          for (ax1_1: int32, 0, 16) {
            block([1, 1, 16, 16], "B_shared_warp") as [v0_3, v1_3, v2_3, v3_3] {
              bind(v0_3, 0)
              bind(v1_3, 0)
              bind(v2_3, ax0_1)
              bind(v3_3, ax1_1)
              tir.reads([B_shared[v0_3, v1_3, v2_3, v3_3]])
              tir.writes([B_shared_warp[v0_3, v1_3, ((v2_3*2) + floordiv(v3_3, 8)), floormod(v3_3, 8)]])
              B_shared_warp[v0_3, v1_3, ((v2_3*2) + floordiv(v3_3, 8)), floormod(v3_3, 8)] = B_shared[v0_3, v1_3, v2_3, v3_3]
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
          attr [IterVar(tx: int32, [0:32], "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
          @tir.mma_fill(8, C_warp_2: Pointer(warp float16), elem_offset: int32, dtype=float16)
        for (kk: int32, 0, 1) {
          for (i: int32, 0, 16) {
            for (j: int32, 0, 16) {
              for (k: int32, 0, 16) {
                block([1, 1, tir.reduce_axis(0, 1), 16, 16, tir.reduce_axis(0, 16)], "B_update") as [vii_1, vjj_1, vkk, vi, vj, vk] {
                  bind(vii_1, ii)
                  bind(vjj_1, jj)
                  bind(vkk, kk)
                  bind(vi, i)
                  bind(vj, j)
                  bind(vk, k)
                  tir.reads([C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))], A_shared_warp[vii_1, vkk, ((vi*2) + floordiv(vk, 8)), floormod(vk, 8)], B_shared_warp[vjj_1, vkk, ((vj*2) + floordiv(vk, 8)), floormod(vk, 8)]])
                  tir.writes([C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))]])
                  C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))] = (C_warp[vii_1, vjj_1, ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(vj, 8)*4) + (floordiv(vi, 8)*2)) + floormod(vj, 2))] + (A_shared_warp[vii_1, vkk, ((vi*2) + floordiv(vk, 8)), floormod(vk, 8)]*B_shared_warp[vjj_1, vkk, ((vj*2) + floordiv(vk, 8)), floormod(vk, 8)]))
              }
            }
          }
        }
        for (ax0_2: int32, 0, 16) {
          for (ax1_2: int32, 0, 16) {
            block([1, 1, 16, 16], "C_warp") as [v0_4, v1_4, v2_4, v3_4] {
              bind(v0_4, 0)
              bind(v1_4, 0)
              bind(v2_4, ax0_2)
              bind(v3_4, ax1_2)
              tir.reads([C_warp[v0_4, v1_4, ((floormod(v2_4, 8)*4) + floordiv(floormod(v3_4, 8), 2)), (((floordiv(v3_4, 8)*4) + (floordiv(v2_4, 8)*2)) + floormod(v3_4, 2))]])
              tir.writes([C[v0_4, v1_4, v2_4, v3_4]])
              C[v0_4, v1_4, v2_4, v3_4] = C_warp[v0_4, v1_4, ((floormod(v2_4, 8)*4) + floordiv(floormod(v3_4, 8), 2)), (((floordiv(v3_4, 8)*4) + (floordiv(v2_4, 8)*2)) + floormod(v3_4, 2))]
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