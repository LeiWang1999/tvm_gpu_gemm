#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [8192, 8192], []),
             B: Buffer(B_1: Pointer(global float16), float16, [8192, 8192], []),
             C: Buffer(C_1: Pointer(global float16), float16, [8192, 8192], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_global = alloc_buffer(float16[512, 512, 16, 16])
    A_global_shared = alloc_buffer(float16[512, 512, 16, 16])
    A_global_shared_warp = alloc_buffer(float16[512, 512, 16, 16])
    B_global = alloc_buffer(float16[512, 512, 16, 16])
    B_global_shared = alloc_buffer(float16[512, 512, 16, 16])
    B_global_shared_warp = alloc_buffer(float16[512, 512, 16, 16])
    C_warp = alloc_buffer(float16[512, 512, 16, 16])
     {
      for (ax0: int32, 0, 8192) {
        for (ax1: int32, 0, 8192) {
          block([8192, 8192], "B_global") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([B[v0, v1]])
            tir.writes([B_global[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)]])
            B_global[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)] = B[v0, v1]
        }
      }
      for (ax0_1: int32, 0, 8192) {
        for (ax1_1: int32, 0, 8192) {
          block([8192, 8192], "A_global") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([A[v0_1, v1_1]])
            tir.writes([A_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]])
            A_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)] = A[v0_1, v1_1]
        }
      }
      for (ax0_2: int32, 0, 8192) {
        for (ax1_2: int32, 0, 8192) {
          block([8192, 8192], "A_global_shared") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([A_global[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
            tir.writes([A_global_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
            A_global_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)] = A_global[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]
        }
      }
      for (ax0_3: int32, 0, 8192) {
        for (ax1_3: int32, 0, 8192) {
          block([8192, 8192], "A_global_shared_warp") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([A_global_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
            tir.writes([A_global_shared_warp[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
            A_global_shared_warp[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)] = A_global_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]
        }
      }
      for (ax0_4: int32, 0, 8192) {
        for (ax1_4: int32, 0, 8192) {
          block([8192, 8192], "B_global_shared") as [v0_4, v1_4] {
            bind(v0_4, ax0_4)
            bind(v1_4, ax1_4)
            tir.reads([B_global[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
            tir.writes([B_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
            B_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)] = B_global[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]
        }
      }
      for (ax0_5: int32, 0, 8192) {
        for (ax1_5: int32, 0, 8192) {
          block([8192, 8192], "B_global_shared_warp") as [v0_5, v1_5] {
            bind(v0_5, ax0_5)
            bind(v1_5, ax1_5)
            tir.reads([B_global_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
            tir.writes([B_global_shared_warp[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
            B_global_shared_warp[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)] = B_global_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]
        }
      }
      for (i: int32, 0, 8192) {
        for (j: int32, 0, 8192) {
          for (k: int32, 0, 8192) {
            block([8192, 8192, tir.reduce_axis(0, 8192)], "B") as [vi, vj, vk] {
              bind(vi, i)
              bind(vj, j)
              bind(vk, k)
              tir.reads([A_global_shared_warp[floordiv(vi, 16), floordiv(vk, 16), floormod(vi, 16), floormod(vk, 16)], B_global_shared_warp[floordiv(vk, 16), floordiv(vj, 16), floormod(vk, 16), floormod(vj, 16)]])
              tir.writes([C_warp[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)]])
              with init() {
                C_warp[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] = 0f16
              }
              C_warp[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] = (C_warp[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] + (A_global_shared_warp[floordiv(vi, 16), floordiv(vk, 16), floormod(vi, 16), floormod(vk, 16)]*B_global_shared_warp[floordiv(vk, 16), floordiv(vj, 16), floormod(vk, 16), floormod(vj, 16)]))
          }
        }
      }
      for (ax0_6: int32, 0, 8192) {
        for (ax1_6: int32, 0, 8192) {
          block([8192, 8192], "C_warp") as [v0_6, v1_6] {
            bind(v0_6, ax0_6)
            bind(v1_6, ax1_6)
            tir.reads([C_warp[floordiv(v0_6, 16), floordiv(v1_6, 16), floormod(v0_6, 16), floormod(v1_6, 16)]])
            tir.writes([C[v0_6, v1_6]])
            C[v0_6, v1_6] = C_warp[floordiv(v0_6, 16), floordiv(v1_6, 16), floormod(v0_6, 16), floormod(v1_6, 16)]
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