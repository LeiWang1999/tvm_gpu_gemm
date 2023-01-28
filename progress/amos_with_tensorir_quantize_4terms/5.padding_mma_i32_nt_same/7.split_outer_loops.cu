#[version = "0.0.5"]
@main = primfn(A_handle: handle, B_handle: handle, C_handle: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global int8), int8, [16384, 16384], []),
             C: Buffer(C_1: Pointer(global int32), int32, [16384, 16384], [])}
  buffer_map = {A_handle: A, B_handle: B, C_handle: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    PA = alloc_buffer(int32[16384])
    for (i_0: int32, 0, 128) "thread_binding" {
      for (j_0: int32, 0, 64) "thread_binding" {
        for (k_0: int32, 0, 256) {
          for (i_1_0: int32, 0, 2) {
            for (j_1_0: int32, 0, 4) {
              for (i_1_1: int32, 0, 64) {
                for (j_1_1: int32, 0, 64) {
                  for (k_1: int32, 0, 64) {
                    block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi, vj, vk] {
                      bind(vi, (((i_0*128) + (i_1_0*64)) + i_1_1))
                      bind(vj, (((j_0*256) + (j_1_0*64)) + j_1_1))
                      bind(vk, ((k_0*64) + k_1))
                      tir.reads([A[vi, vk], B[vj, vk]])
                      tir.writes([C[vi, vj]])
                      with init() {
                        C[vi, vj] = 0
                      }
                      C[vi, vj] = (C[vi, vj] + (cast(int32, A[vi, vk])*cast(int32, B[vj, vk])))
                  }
                }
              }
            }
          }
          for (ax0: int32, 0, 128) {
            for (ax1: int32, 0, 64) {
              block([16384, tir.reduce_axis(0, 16384)], "Pre_compute_A") as [vi_1, vk_1] {
                bind(vi_1, ((i_0*128) + ax0))
                bind(vk_1, ((k_0*64) + ax1))
                tir.reads([A[vi_1, vk_1]])
                tir.writes([PA[vi_1]])
                with init() {
                  PA[vi_1] = 0
                }
                PA[vi_1] = (PA[vi_1] + (1*cast(int32, A[vi_1, vk_1])))
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