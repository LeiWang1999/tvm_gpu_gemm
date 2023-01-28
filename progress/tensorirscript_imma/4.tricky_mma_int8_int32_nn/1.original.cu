#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [16, 8, 16, 32], []),
             B: Buffer(B_1: Pointer(global int8), int8, [8, 16, 32, 16], []),
             C: Buffer(C_1: Pointer(global int32), int32, [16, 16, 16, 16], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (ii: int32, 0, 16) {
      for (jj: int32, 0, 16) {
        for (kk: int32, 0, 8) {
          for (i: int32, 0, 16) {
            for (j: int32, 0, 16) {
              for (k: int32, 0, 32) {
                block([16, 16, tir.reduce_axis(0, 8), 16, 16, tir.reduce_axis(0, 32)], "B") as [vii, vjj, vkk, vi, vj, vk] {
                  bind(vii, ii)
                  bind(vjj, jj)
                  bind(vkk, kk)
                  bind(vi, i)
                  bind(vj, j)
                  bind(vk, k)
                  tir.reads([A[vii, vkk, vi, vk], B[vkk, vjj, vk, vj]])
                  tir.writes([C[vii, vjj, vi, vj]])
                  with init() {
                    C[vii, vjj, vi, vj] = 0
                  }
                  C[vii, vjj, vi, vj] = (C[vii, vjj, vi, vj] + (cast(int32, A[vii, vkk, vi, vk])*cast(int32, B[vkk, vjj, vk, vj])))
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