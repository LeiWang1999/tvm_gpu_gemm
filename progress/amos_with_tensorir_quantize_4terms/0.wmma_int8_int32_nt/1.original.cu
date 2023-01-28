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
    PA = alloc_buffer(int32[16384])
    PB = alloc_buffer(int32[16384])
    QC = alloc_buffer(int32[16384, 16384])
     {
      for (i: int32, 0, 16384) {
        for (j: int32, 0, 16384) {
          for (k: int32, 0, 16384) {
            block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi, vj, vk] {
              bind(vi, i)
              bind(vj, j)
              bind(vk, k)
              tir.reads([A[vi, vk], B[vj, vk]])
              tir.writes([QC[vi, vj]])
              with init() {
                QC[vi, vj] = 0
              }
              QC[vi, vj] = (QC[vi, vj] + (cast(int32, A[vi, vk])*cast(int32, B[vj, vk])))
          }
        }
      }
      for (i_1: int32, 0, 16384) {
        for (k_1: int32, 0, 16384) {
          block([16384, tir.reduce_axis(0, 16384)], "Pre_compute_A") as [vi_1, vk_1] {
            bind(vi_1, i_1)
            bind(vk_1, k_1)
            tir.reads([A[vi_1, vk_1]])
            tir.writes([PA[vi_1]])
            with init() {
              PA[vi_1] = 0
            }
            PA[vi_1] = (PA[vi_1] + (4*cast(int32, A[vi_1, vk_1])))
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