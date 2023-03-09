#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [18966528, 32], []),
             B: Buffer(B_1: Pointer(global int8), int8, [1, 32], []),
             C: Buffer(C_1: Pointer(global int32), int32, [18966528, 1], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_local = alloc_buffer(int8[18966528, 32])
    B_local = alloc_buffer(int8[1, 32])
     {
      for (ax0: int32, 0, 1) {
        for (ax1: int32, 0, 32) {
          block([1, 32], "B_local") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([B[v0, v1]])
            tir.writes([B_local[v0, v1]])
            B_local[v0, v1] = B[v0, v1]
        }
      }
      for (ax0_1: int32, 0, 18966528) {
        for (ax1_1: int32, 0, 32) {
          block([18966528, 32], "A_local") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([A[v0_1, v1_1]])
            tir.writes([A_local[v0_1, v1_1]])
            A_local[v0_1, v1_1] = A[v0_1, v1_1]
        }
      }
      for (i: int32, 0, 18966528) {
        for (j: int32, 0, 1) {
          for (k: int32, 0, 32) {
            block([18966528, 1, tir.reduce_axis(0, 32)], "B") as [vi, vj, vk] {
              bind(vi, i)
              bind(vj, j)
              bind(vk, k)
              tir.reads([A_local[vi, vk], B_local[vj, vk]])
              tir.writes([C[vi, vj]])
              with init() {
                C[vi, vj] = 0
              }
              C[vi, vj] = (C[vi, vj] + (cast(int32, A_local[vi, vk])*cast(int32, B_local[vj, vk])))
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