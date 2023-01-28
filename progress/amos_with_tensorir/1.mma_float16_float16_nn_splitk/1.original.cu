#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [1024, 16384], []),
             B: Buffer(B_1: Pointer(global float16), float16, [16384, 1024], []),
             C: Buffer(C_1: Pointer(global float16), float16, [1024, 1024], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    TC = alloc_buffer(float16[1, 1024, 1024])
     {
      for (sk: int32, 0, 1) {
        for (i: int32, 0, 1024) {
          for (j: int32, 0, 1024) {
            for (k: int32, 0, 16384) {
              block([1, 1024, 1024, tir.reduce_axis(0, 16384)], "B") as [vsk, vi, vj, vk] {
                bind(vsk, sk)
                bind(vi, i)
                bind(vj, j)
                bind(vk, k)
                tir.reads([A[vi, ((vsk*16384) + vk)], B[((vsk*16384) + vk), vj]])
                tir.writes([TC[vsk, vi, vj]])
                with init() {
                  TC[vsk, vi, vj] = 0f16
                }
                TC[vsk, vi, vj] = (TC[vsk, vi, vj] + (A[vi, ((vsk*16384) + vk)]*B[((vsk*16384) + vk), vj]))
            }
          }
        }
      }
      for (sk_1: int32, 0, 1) {
        for (i_1: int32, 0, 1024) {
          for (j_1: int32, 0, 1024) {
            block([1, 1024, 1024], "C") as [vsk_1, vi_1, vj_1] {
              bind(vsk_1, sk_1)
              bind(vi_1, i_1)
              bind(vj_1, j_1)
              tir.reads([C[vi_1, vj_1], TC[vsk_1, vi_1, vj_1]])
              tir.writes([])
              @tir.atomic_add(@tir.address_of(C[vi_1, vj_1], dtype=handle), TC[vsk_1, vi_1, vj_1], dtype=float16)
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