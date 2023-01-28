#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global float16), float16, [16384, 16384], []),
             C: Buffer(C_1: Pointer(global float16), float16, [16384, 16384], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i: int32, 0, 16384) {
      for (j: int32, 0, 16384) {
        for (k: int32, 0, 16384) {
          block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi, vj, vk] {
            bind(vi, i)
            bind(vj, j)
            bind(vk, k)
            tir.reads([A[vi, vk], B[vk, vj]])
            tir.writes([C[vi, vj]])
            with init() {
              C[vi, vj] = 0f16
            }
            C[vi, vj] = (C[vi, vj] + (A[vi, vk]*B[vk, vj]))
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