#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [256, 256], []),
             B: Buffer(B_1: Pointer(global int8), int8, [256, 256], []),
             C: Buffer(C_1: Pointer(global int32), int32, [256, 256], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i_0: int32, 0, 2) {
      for (j_0: int32, 0, 1) {
        for (k_0: int32, 0, 4) {
          for (i_1: int32, 0, 128) {
            for (j_1: int32, 0, 256) {
              for (k_1: int32, 0, 64) {
                block([256, 256, tir.reduce_axis(0, 256)], "B") as [vi, vj, vk] {
                  bind(vi, ((i_0*128) + i_1))
                  bind(vj, ((j_0*256) + j_1))
                  bind(vk, ((k_0*64) + k_1))
                  tir.reads([A[vi, vk], B[vk, vj]])
                  tir.writes([C[vi, vj]])
                  with init() {
                    C[vi, vj] = 0
                  }
                  C[vi, vj] = (C[vi, vj] + (cast(int32, A[vi, vk])*cast(int32, B[vk, vj])))
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