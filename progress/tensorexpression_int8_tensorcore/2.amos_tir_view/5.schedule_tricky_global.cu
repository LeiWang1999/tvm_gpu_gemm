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
    A_global = alloc_buffer(int8[16384, 16384])
    A_global_shared = alloc_buffer(int8[16384, 16384])
    B_global = alloc_buffer(int8[16384, 16384])
    B_global_shared = alloc_buffer(int8[16384, 16384])
    C_global = alloc_buffer(int32[16384, 16384])
     {
      for (ax0_0: int32, 0, 1024) {
        for (ax1_0: int32, 0, 1024) {
          for (ax0_1: int32, 0, 16) {
            for (ax1_1: int32, 0, 16) {
              block([16384, 16384], "B_global") as [v0, v1] {
                bind(v0, ((ax0_0*16) + ax0_1))
                bind(v1, ((ax1_0*16) + ax1_1))
                tir.reads([B[v0, v1]])
                tir.writes([B_global[v0, v1]])
                B_global[v0, v1] = B[v0, v1]
            }
          }
        }
      }
      for (ax0_0_1: int32, 0, 1024) {
        for (ax1_0_1: int32, 0, 1024) {
          for (ax0_1_1: int32, 0, 16) {
            for (ax1_1_1: int32, 0, 16) {
              block([16384, 16384], "A_global") as [v0_1, v1_1] {
                bind(v0_1, ((ax0_0_1*16) + ax0_1_1))
                bind(v1_1, ((ax1_0_1*16) + ax1_1_1))
                tir.reads([A[v0_1, v1_1]])
                tir.writes([A_global[v0_1, v1_1]])
                A_global[v0_1, v1_1] = A[v0_1, v1_1]
            }
          }
        }
      }
      for (ax0: int32, 0, 16384) {
        for (ax1: int32, 0, 16384) {
          block([16384, 16384], "A_global_shared") as [v0_2, v1_2] {
            bind(v0_2, ax0)
            bind(v1_2, ax1)
            tir.reads([A_global[v0_2, v1_2]])
            tir.writes([A_global_shared[v0_2, v1_2]])
            A_global_shared[v0_2, v1_2] = A_global[v0_2, v1_2]
        }
      }
      for (ax0_2: int32, 0, 16384) {
        for (ax1_2: int32, 0, 16384) {
          block([16384, 16384], "B_global_shared") as [v0_3, v1_3] {
            bind(v0_3, ax0_2)
            bind(v1_3, ax1_2)
            tir.reads([B_global[v0_3, v1_3]])
            tir.writes([B_global_shared[v0_3, v1_3]])
            B_global_shared[v0_3, v1_3] = B_global[v0_3, v1_3]
        }
      }
      for (i: int32, 0, 16384) {
        for (j: int32, 0, 16384) {
          for (k: int32, 0, 16384) {
            block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi, vj, vk] {
              bind(vi, i)
              bind(vj, j)
              bind(vk, k)
              tir.reads([A_global_shared[vi, vk], B_global_shared[vj, vk]])
              tir.writes([C_global[vi, vj]])
              with init() {
                C_global[vi, vj] = 0
              }
              C_global[vi, vj] = (C_global[vi, vj] + (cast(int32, A_global_shared[vi, vk])*cast(int32, B_global_shared[vj, vk])))
          }
        }
      }
      for (ax0_3: int32, 0, 16384) {
        for (ax1_3: int32, 0, 16384) {
          block([16384, 16384], "C_global") as [v0_4, v1_4] {
            bind(v0_4, ax0_3)
            bind(v1_4, ax1_3)
            tir.reads([C_global[v0_4, v1_4]])
            tir.writes([C[v0_4, v1_4]])
            C[v0_4, v1_4] = C_global[v0_4, v1_4]
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