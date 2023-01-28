#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [256, 256], []),
             B: Buffer(B_1: Pointer(global float16), float16, [256, 256], []),
             C: Buffer(C_1: Pointer(global float16), float16, [256, 256], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i_0: int32, 0, 1) {
      for (i_1: int32, 0, 256) {
        for (j_0: int32, 0, 2) {
          for (j_1: int32, 0, 128) {
            for (k_0: int32, 0, 8) {
              for (k_1: int32, 0, 32) {
                block([256, 256, tir.reduce_axis(0, 256)], "B") as [vi, vj, vk] {
                  bind(vi, ((i_0*256) + i_1))
                  bind(vj, ((j_0*128) + j_1))
                  bind(vk, ((k_0*32) + k_1))
                  tir.reads([A[vi, vk], B[vj, vk]])
                  tir.writes([C[vi, vj]])
                  with init() {
                    C[vi, vj] = 0f16
                  }
                  C[vi, vj] = (C[vi, vj] + (A[vi, vk]*B[vj, vk]))
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