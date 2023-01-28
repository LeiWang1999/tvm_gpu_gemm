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
    for (ii: int32, 0, 1024) {
      for (jj: int32, 0, 1024) {
        for (kk: int32, 0, 1024) {
          for (i: int32, 0, 16) {
            for (j: int32, 0, 16) {
              for (k: int32, 0, 16) {
                block([1024, 1024, tir.reduce_axis(0, 1024), 16, 16, tir.reduce_axis(0, 16)], "B") as [vii, vjj, vkk, vi, vj, vk] {
                  bind(vii, ii)
                  bind(vjj, jj)
                  bind(vkk, kk)
                  bind(vi, i)
                  bind(vj, j)
                  bind(vk, k)
                  tir.reads([A[vii, vkk, vi, vk], B[vjj, vkk, vj, vk]])
                  tir.writes([C[vii, vjj, vi, vj]])
                  with init() {
                    C[vii, vjj, vi, vj] = 0f16
                  }
                  C[vii, vjj, vi, vj] = (C[vii, vjj, vi, vj] + (A[vii, vkk, vi, vk]*B[vjj, vkk, vj, vk]))
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