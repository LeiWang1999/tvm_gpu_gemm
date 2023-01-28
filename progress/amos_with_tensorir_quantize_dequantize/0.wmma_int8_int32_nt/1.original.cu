#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global int8), int8, [16384, 16384], []),
             C: Buffer(C_1: Pointer(global int8), int8, [16384, 16384], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    QA = alloc_buffer(int8[16384, 16384])
    QB = alloc_buffer(int8[16384, 16384])
    QC = alloc_buffer(int32[16384, 16384])
     {
      for (i: int32, 0, 16384) {
        for (j: int32, 0, 16384) {
          block([16384, 16384], "Quantize_A") as [vi, vj] {
            bind(vi, i)
            bind(vj, j)
            tir.reads([A[vi, vj]])
            tir.writes([QA[vi, vj]])
            QA[vi, vj] = cast(int8, (@tir.round((cast(float32, A[vi, vj])*0.5f32), dtype=float32) - 0f32))
        }
      }
      for (i_1: int32, 0, 16384) {
        for (j_1: int32, 0, 16384) {
          block([16384, 16384], "Quantize_B") as [vi_1, vj_1] {
            bind(vi_1, i_1)
            bind(vj_1, j_1)
            tir.reads([B[vi_1, vj_1]])
            tir.writes([QB[vi_1, vj_1]])
            QB[vi_1, vj_1] = cast(int8, (@tir.round((cast(float32, B[vi_1, vj_1])*0.1f32), dtype=float32) - 0f32))
        }
      }
      for (i_2: int32, 0, 16384) {
        for (j_2: int32, 0, 16384) {
          for (k: int32, 0, 16384) {
            block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi_2, vj_2, vk] {
              bind(vi_2, i_2)
              bind(vj_2, j_2)
              bind(vk, k)
              tir.reads([QA[vi_2, vk], QB[vj_2, vk]])
              tir.writes([QC[vi_2, vj_2]])
              with init() {
                QC[vi_2, vj_2] = 0
              }
              QC[vi_2, vj_2] = (QC[vi_2, vj_2] + (cast(int32, QA[vi_2, vk])*cast(int32, QB[vj_2, vk])))
          }
        }
      }
      for (i_3: int32, 0, 16384) {
        for (j_3: int32, 0, 16384) {
          block([16384, 16384], "DeQuantize_C") as [vi_3, vj_3] {
            bind(vi_3, i_3)
            bind(vj_3, j_3)
            tir.reads([QC[vi_3, vj_3]])
            tir.writes([C[vi_3, vj_3]])
            C[vi_3, vj_3] = cast(int8, ((cast(float32, QC[vi_3, vj_3]) / 0.01f32) + 0f32))
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