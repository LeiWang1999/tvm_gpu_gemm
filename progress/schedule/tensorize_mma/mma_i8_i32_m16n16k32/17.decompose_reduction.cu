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
    A_shared_warp = alloc_buffer(int8[1, 1, 16, 32])
    B_shared = alloc_buffer(int8[1, 1, 16, 32])
    B_shared_warp = alloc_buffer(int8[1, 1, 16, 32])
    B_shared_warp_warp = alloc_buffer(int8[1, 1, 16, 32])
    C_warp = alloc_buffer(int32[1, 1, 16, 16])
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
        for (ax0: int32, 0, 16) {
          for (ax1: int32, 0, 32) {
            block([1, 1, 16, 32], "A_shared_warp") as [v0_1, v1_1, v2_1, v3_1] {
              bind(v0_1, 0)
              bind(v1_1, 0)
              bind(v2_1, ax0)
              bind(v3_1, ax1)
              tir.reads([A_shared[v0_1, v1_1, v2_1, v3_1]])
              tir.writes([A_shared_warp[v0_1, v1_1, v2_1, v3_1]])
              A_shared_warp[v0_1, v1_1, v2_1, v3_1] = A_shared[v0_1, v1_1, v2_1, v3_1]
          }
        }
        for (ax0_ax1_fused_0_1: int32, 0, 1) "thread_binding" {
          for (ax0_ax1_fused_1_1: int32, 0, 1) "thread_binding" {
            for (ax0_ax1_fused_2_1: int32, 0, 1) {
              for (ax0_ax1_fused_3_1: int32, 0, 32) "thread_binding" {
                for (ax0_ax1_fused_4_1: int32, 0, 16) "vectorized" {
                  block([1, 1, 16, 32], "B_shared") as [v0_2, v1_2, v2_2, v3_2] {
                    bind(v0_2, 0)
                    bind(v1_2, 0)
                    bind(v2_2, floordiv((((((ax0_ax1_fused_0_1*512) + (ax0_ax1_fused_1_1*512)) + (ax0_ax1_fused_2_1*512)) + (ax0_ax1_fused_3_1*16)) + ax0_ax1_fused_4_1), 32))
                    bind(v3_2, floormod((((((ax0_ax1_fused_0_1*512) + (ax0_ax1_fused_1_1*512)) + (ax0_ax1_fused_2_1*512)) + (ax0_ax1_fused_3_1*16)) + ax0_ax1_fused_4_1), 32))
                    tir.reads([B[v0_2, v1_2, ((floormod(v2_2, 8)*2) + floordiv(v3_2, 16)), ((floordiv(v2_2, 8)*16) + floormod(v3_2, 16))]])
                    tir.writes([B_shared[v0_2, v1_2, v2_2, v3_2]])
                    B_shared[v0_2, v1_2, v2_2, v3_2] = B[v0_2, v1_2, ((floormod(v2_2, 8)*2) + floordiv(v3_2, 16)), ((floordiv(v2_2, 8)*16) + floormod(v3_2, 16))]
                }
              }
            }
          }
        }
        for (ax0_1: int32, 0, 16) {
          for (ax1_1: int32, 0, 32) {
            block([1, 1, 16, 32], "B_shared_warp") as [v0_3, v1_3, v2_3, v3_3] {
              bind(v0_3, 0)
              bind(v1_3, 0)
              bind(v2_3, ax0_1)
              bind(v3_3, ax1_1)
              tir.reads([B_shared[v0_3, v1_3, v2_3, v3_3]])
              tir.writes([B_shared_warp[v0_3, v1_3, v2_3, v3_3]])
              B_shared_warp[v0_3, v1_3, v2_3, v3_3] = B_shared[v0_3, v1_3, v2_3, v3_3]
          }
        }
        for (ax0_2: int32, 0, 16) {
          for (ax1_2: int32, 0, 32) {
            block([1, 1, 16, 32], "B_shared_warp_warp") as [v0_4, v1_4, v2_4, v3_4] {
              bind(v0_4, 0)
              bind(v1_4, 0)
              bind(v2_4, ax0_2)
              bind(v3_4, ax1_2)
              tir.reads([B_shared_warp[v0_4, v1_4, v2_4, v3_4]])
              tir.writes([B_shared_warp_warp[v0_4, v1_4, v2_4, v3_4]])
              B_shared_warp_warp[v0_4, v1_4, v2_4, v3_4] = B_shared_warp[v0_4, v1_4, v2_4, v3_4]
          }
        }
        for (i_init: int32, 0, 16) {
          for (j_init: int32, 0, 16) {
            block([1, 1, 16, 16], "B_init") as [vii, vjj, vi, vj] {
              bind(vii, ii)
              bind(vjj, jj)
              bind(vi, i_init)
              bind(vj, j_init)
              tir.reads([])
              tir.writes([C_warp[vii, vjj, vi, vj]])
              C_warp[vii, vjj, vi, vj] = 0
          }
        }
        for (kk: int32, 0, 1) {
          for (i: int32, 0, 16) {
            for (j: int32, 0, 16) {
              for (k: int32, 0, 32) {
                block([1, 1, tir.reduce_axis(0, 1), 16, 16, tir.reduce_axis(0, 32)], "B_update") as [vii_1, vjj_1, vkk, vi_1, vj_1, vk] {
                  bind(vii_1, ii)
                  bind(vjj_1, jj)
                  bind(vkk, kk)
                  bind(vi_1, i)
                  bind(vj_1, j)
                  bind(vk, k)
                  tir.reads([C_warp[vii_1, vjj_1, vi_1, vj_1], A_shared_warp[vii_1, vkk, vi_1, vk], B_shared_warp_warp[vjj_1, vkk, vj_1, vk]])
                  tir.writes([C_warp[vii_1, vjj_1, vi_1, vj_1]])
                  C_warp[vii_1, vjj_1, vi_1, vj_1] = (C_warp[vii_1, vjj_1, vi_1, vj_1] + (cast(int32, A_shared_warp[vii_1, vkk, vi_1, vk])*cast(int32, B_shared_warp_warp[vjj_1, vkk, vj_1, vk])))
              }
            }
          }
        }
        for (ax0_3: int32, 0, 16) {
          for (ax1_3: int32, 0, 16) {
            block([1, 1, 16, 16], "C_warp") as [v0_5, v1_5, v2_5, v3_5] {
              bind(v0_5, 0)
              bind(v1_5, 0)
              bind(v2_5, ax0_3)
              bind(v3_5, ax1_3)
              tir.reads([C_warp[v0_5, v1_5, v2_5, v3_5]])
              tir.writes([C[v0_5, v1_5, v2_5, v3_5]])
              C[v0_5, v1_5, v2_5, v3_5] = C_warp[v0_5, v1_5, v2_5, v3_5]
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