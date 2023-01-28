#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [1024, 16384], []),
             B: Buffer(B_1: Pointer(global float16), float16, [16384, 1024], []),
             C: Buffer(C_1: Pointer(global float16), float16, [1024, 1024], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    TC = alloc_buffer(float16[1, 1024, 1024])
    A_global = alloc_buffer(float16[1024, 16384])
    A_global_shared = alloc_buffer(float16[1024, 16384])
    A_global_shared_warp = alloc_buffer(float16[1024, 16384])
    B_global = alloc_buffer(float16[16384, 1024])
    B_global_shared = alloc_buffer(float16[16384, 1024])
    B_global_shared_warp = alloc_buffer(float16[16384, 1024])
    TC_warp = alloc_buffer(float16[1, 1024, 1024])
     {
      for (ax0: int32, 0, 16384) {
        for (ax1: int32, 0, 1024) {
          block([16384, 1024], "B_global") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([B[v0, v1]])
            tir.writes([B_global[v0, v1]])
            B_global[v0, v1] = B[v0, v1]
        }
      }
      for (ax0_1: int32, 0, 1024) {
        for (ax1_1: int32, 0, 16384) {
          block([1024, 16384], "A_global") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([A[v0_1, v1_1]])
            tir.writes([A_global[v0_1, v1_1]])
            A_global[v0_1, v1_1] = A[v0_1, v1_1]
        }
      }
      for (ax0_2: int32, 0, 1024) {
        for (ax1_2: int32, 0, 16384) {
          block([1024, 16384], "A_global_shared") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([A_global[v0_2, v1_2]])
            tir.writes([A_global_shared[v0_2, v1_2]])
            A_global_shared[v0_2, v1_2] = A_global[v0_2, v1_2]
        }
      }
      for (ax0_3: int32, 0, 1024) {
        for (ax1_3: int32, 0, 16384) {
          block([1024, 16384], "A_global_shared_warp") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([A_global_shared[v0_3, v1_3]])
            tir.writes([A_global_shared_warp[v0_3, v1_3]])
            A_global_shared_warp[v0_3, v1_3] = A_global_shared[v0_3, v1_3]
        }
      }
      for (ax0_4: int32, 0, 16384) {
        for (ax1_4: int32, 0, 1024) {
          block([16384, 1024], "B_global_shared") as [v0_4, v1_4] {
            bind(v0_4, ax0_4)
            bind(v1_4, ax1_4)
            tir.reads([B_global[v0_4, v1_4]])
            tir.writes([B_global_shared[v0_4, v1_4]])
            B_global_shared[v0_4, v1_4] = B_global[v0_4, v1_4]
        }
      }
      for (ax0_5: int32, 0, 16384) {
        for (ax1_5: int32, 0, 1024) {
          block([16384, 1024], "B_global_shared_warp") as [v0_5, v1_5] {
            bind(v0_5, ax0_5)
            bind(v1_5, ax1_5)
            tir.reads([B_global_shared[v0_5, v1_5]])
            tir.writes([B_global_shared_warp[v0_5, v1_5]])
            B_global_shared_warp[v0_5, v1_5] = B_global_shared[v0_5, v1_5]
        }
      }
      for (sk: int32, 0, 1) {
        for (i: int32, 0, 1024) {
          for (j: int32, 0, 1024) {
            for (k: int32, 0, 16384) {
              block([1, 1024, 1024, tir.reduce_axis(0, 16384)], "B") as [vsk, vi, vj, vk] {
                bind(vsk, sk)
                bind(vi, i)
                bind(vj, j)
                bind(vk, k)
                tir.reads([A_global_shared_warp[vi, ((vsk*16384) + vk)], B_global_shared_warp[((vsk*16384) + vk), vj]])
                tir.writes([TC_warp[vsk, vi, vj]])
                with init() {
                  TC_warp[vsk, vi, vj] = 0f16
                }
                TC_warp[vsk, vi, vj] = (TC_warp[vsk, vi, vj] + (A_global_shared_warp[vi, ((vsk*16384) + vk)]*B_global_shared_warp[((vsk*16384) + vk), vj]))
            }
          }
        }
      }
      for (ax0_6: int32, 0, 1) {
        for (ax1_6: int32, 0, 1024) {
          for (ax2: int32, 0, 1024) {
            block([1, 1024, 1024], "TC_warp") as [v0_6, v1_6, v2] {
              bind(v0_6, ax0_6)
              bind(v1_6, ax1_6)
              bind(v2, ax2)
              tir.reads([TC_warp[v0_6, v1_6, v2]])
              tir.writes([TC[v0_6, v1_6, v2]])
              TC[v0_6, v1_6, v2] = TC_warp[v0_6, v1_6, v2]
          }
        }
      }
      for (sk_1: int32, 0, 1) {
        for (i_1: int32, 0, 1024) {
          for (j_1: int32, 0, 1024) {
            block([1, 1024, 1024], "C") as [vsk_1, vi_1, vj_1] {
              bind(vsk_1, sk_1)
              bind(vi_1, i_1)
              bind(vj_1, j_1)
              tir.reads([C[vi_1, vj_1], TC[vsk_1, vi_1, vj_1]])
              tir.writes([])
              @tir.atomic_add(@tir.address_of(C[vi_1, vj_1], dtype=handle), TC[vsk_1, vi_1, vj_1], dtype=float16)
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