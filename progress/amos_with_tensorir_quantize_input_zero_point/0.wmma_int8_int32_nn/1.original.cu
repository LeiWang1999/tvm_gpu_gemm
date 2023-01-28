#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle, pb: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global int8), int8, [16384, 16384], []),
             C: Buffer(C_1: Pointer(global int32), int32, [16384, 16384], []),
             PB: Buffer(PB_1: Pointer(global int32), int32, [16384], [])}
  buffer_map = {a: A, b: B, c: C, pb: PB} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    QC = alloc_buffer(int32[16384, 16384])
     {
      for (i: int32, 0, 16384) {
        for (j: int32, 0, 16384) {
          for (k: int32, 0, 16384) {
            block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi, vj, vk] {
              bind(vi, i)
              bind(vj, j)
              bind(vk, k)
              tir.reads([A[vi, vk], B[vk, vj]])
              tir.writes([QC[vi, vj]])
              with init() {
                QC[vi, vj] = 0
              }
              QC[vi, vj] = (QC[vi, vj] + (cast(int32, A[vi, vk])*cast(int32, B[vk, vj])))
          }
        }
      }
      for (i_1: int32, 0, 16384) {
        for (j_1: int32, 0, 16384) {
          block([16384, 16384], "C") as [vi_1, vj_1] {
            bind(vi_1, i_1)
            bind(vj_1, j_1)
            tir.reads([QC[vi_1, vj_1], PB[vj_1]])
            tir.writes([C[vi_1, vj_1]])
            C[vi_1, vj_1] = ((QC[vi_1, vj_1] + 12) + PB[vj_1])
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