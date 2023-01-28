#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [196, 36, 16, 16], []),
             B: Buffer(B_1: Pointer(global float16), float16, [36, 4, 16, 16], []),
             C: Buffer(C_1: Pointer(global float16), float16, [1, 196, 4, 16, 16], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (sk: int32, 0, 1) {
      for (ii: int32, 0, 196) {
        for (jj: int32, 0, 4) {
          for (kk: int32, 0, 36) {
            for (i: int32, 0, 16) {
              for (j: int32, 0, 16) {
                for (k: int32, 0, 16) {
                  block([1, 196, 4, tir.reduce_axis(0, 36), 16, 16, tir.reduce_axis(0, 16)], "B") as [vsk, vii, vjj, vkk, vi, vj, vk] {
                    bind(vsk, sk)
                    bind(vii, ii)
                    bind(vjj, jj)
                    bind(vkk, kk)
                    bind(vi, i)
                    bind(vj, j)
                    bind(vk, k)
                    tir.reads([A[vii, vkk, vi, vk], B[vkk, vjj, vk, vj]])
                    tir.writes([C[vsk, vii, vjj, vi, vj]])
                    with init() {
                      C[vsk, vii, vjj, vi, vj] = 0f32
                    }
                    C[vsk, vii, vjj, vi, vj] = (C[vsk, vii, vjj, vi, vj] + (A[vii, vkk, vi, vk]*B[vkk, vjj, vk, vj]))
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