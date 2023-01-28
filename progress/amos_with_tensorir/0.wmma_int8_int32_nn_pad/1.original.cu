#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [3136, 576], []),
             B: Buffer(B_1: Pointer(global int8), int8, [576, 64], []),
             C: Buffer(C_1: Pointer(global int32), int32, [3136, 64], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    APad = alloc_buffer(int8[3200, 640])
    BPad = alloc_buffer(int8[640, 128])
    CPad = alloc_buffer(int32[3200, 128])
     {
      for (i: int32, 0, 3200) {
        for (k: int32, 0, 640) {
          block([3200, 640], "APad") as [vi, vk] {
            bind(vi, i)
            bind(vk, k)
            tir.reads([A[vi, vk]])
            tir.writes([APad[vi, vk]])
            APad[vi, vk] = @tir.if_then_else(((vi < 3136) && (vk < 576)), A[vi, vk], 0i8, dtype=int8)
        }
      }
      for (k_1: int32, 0, 640) {
        for (j: int32, 0, 128) {
          block([640, 128], "BPad") as [vk_1, vj] {
            bind(vk_1, k_1)
            bind(vj, j)
            tir.reads([B[vk_1, vj]])
            tir.writes([BPad[vk_1, vj]])
            BPad[vk_1, vj] = @tir.if_then_else(((vk_1 < 576) && (vj < 64)), B[vk_1, vj], 0i8, dtype=int8)
        }
      }
      for (i_1: int32, 0, 3200) {
        for (j_1: int32, 0, 128) {
          for (k_2: int32, 0, 640) {
            block([3200, 128, tir.reduce_axis(0, 640)], "B") as [vi_1, vj_1, vk_2] {
              bind(vi_1, i_1)
              bind(vj_1, j_1)
              bind(vk_2, k_2)
              tir.reads([APad[vi_1, vk_2], BPad[vk_2, vj_1]])
              tir.writes([CPad[vi_1, vj_1]])
              with init() {
                CPad[vi_1, vj_1] = 0
              }
              CPad[vi_1, vj_1] = (CPad[vi_1, vj_1] + (cast(int32, APad[vi_1, vk_2])*cast(int32, BPad[vk_2, vj_1])))
          }
        }
      }
      for (i_2: int32, 0, 3136) {
        for (j_2: int32, 0, 64) {
          block([3136, 64], "CPad") as [vi_2, vj_2] {
            bind(vi_2, i_2)
            bind(vj_2, j_2)
            tir.reads([CPad[vi_2, vj_2]])
            tir.writes([C[vi_2, vj_2]])
            C[vi_2, vj_2] = CPad[vi_2, vj_2]
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