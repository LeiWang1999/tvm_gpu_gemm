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
    APad_global = alloc_buffer(int8[200, 40, 16, 16])
    APad_global_shared = alloc_buffer(int8[200, 40, 16, 16])
    APad_global_shared_wmma.matrix_a = alloc_buffer(int8[200, 40, 16, 16])
    BPad_global = alloc_buffer(int8[40, 8, 16, 16])
    BPad_global_shared = alloc_buffer(int8[40, 8, 16, 16])
    BPad_global_shared_wmma.matrix_b = alloc_buffer(int8[40, 8, 16, 16])
    CPad_shared = alloc_buffer(int32[3200, 128])
    CPad_shared_wmma.accumulator = alloc_buffer(int32[200, 8, 16, 16])
     {
      for (ax0: int32, 0, 3200) {
        for (ax1: int32, 0, 640) {
          block([3200, 640], "APad_global") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([A[v0, v1]])
            tir.writes([APad_global[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)]])
            APad_global[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)] = @tir.if_then_else(((v0 < 3136) && (v1 < 576)), A[v0, v1], 0i8, dtype=int8)
        }
      }
      for (ax0_1: int32, 0, 3200) {
        for (ax1_1: int32, 0, 640) {
          block([3200, 640], "APad_global_shared") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([APad_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]])
            tir.writes([APad_global_shared[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]])
            APad_global_shared[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)] = APad_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]
        }
      }
      for (ax0_2: int32, 0, 3200) {
        for (ax1_2: int32, 0, 640) {
          block([3200, 640], "APad_global_shared_wmma.matrix_a") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([APad_global_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
            tir.writes([APad_global_shared_wmma.matrix_a[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
            APad_global_shared_wmma.matrix_a[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)] = APad_global_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]
        }
      }
      for (ax0_3: int32, 0, 640) {
        for (ax1_3: int32, 0, 128) {
          block([640, 128], "BPad_global") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([B[v0_3, v1_3]])
            tir.writes([BPad_global[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
            BPad_global[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)] = @tir.if_then_else(((v0_3 < 576) && (v1_3 < 64)), B[v0_3, v1_3], 0i8, dtype=int8)
        }
      }
      for (ax0_4: int32, 0, 640) {
        for (ax1_4: int32, 0, 128) {
          block([640, 128], "BPad_global_shared") as [v0_4, v1_4] {
            bind(v0_4, ax0_4)
            bind(v1_4, ax1_4)
            tir.reads([BPad_global[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
            tir.writes([BPad_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
            BPad_global_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)] = BPad_global[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]
        }
      }
      for (ax0_5: int32, 0, 640) {
        for (ax1_5: int32, 0, 128) {
          block([640, 128], "BPad_global_shared_wmma.matrix_b") as [v0_5, v1_5] {
            bind(v0_5, ax0_5)
            bind(v1_5, ax1_5)
            tir.reads([BPad_global_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
            tir.writes([BPad_global_shared_wmma.matrix_b[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
            BPad_global_shared_wmma.matrix_b[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)] = BPad_global_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]
        }
      }
      for (i_0: int32, 0, 200) {
        for (j_0: int32, 0, 8) {
          for (k_0: int32, 0, 40) {
            for (i_1: int32, 0, 16) {
              for (j_1: int32, 0, 16) {
                for (k_1: int32, 0, 16) {
                  block([3200, 128, tir.reduce_axis(0, 640)], "B") as [vi, vj, vk] {
                    bind(vi, ((i_0*16) + i_1))
                    bind(vj, ((j_0*16) + j_1))
                    bind(vk, ((k_0*16) + k_1))
                    tir.reads([APad_global_shared_wmma.matrix_a[floordiv(vi, 16), floordiv(vk, 16), floormod(vi, 16), floormod(vk, 16)], BPad_global_shared_wmma.matrix_b[floordiv(vk, 16), floordiv(vj, 16), floormod(vk, 16), floormod(vj, 16)]])
                    tir.writes([CPad_shared_wmma.accumulator[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)]])
                    with init() {
                      CPad_shared_wmma.accumulator[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] = 0
                    }
                    CPad_shared_wmma.accumulator[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] = (CPad_shared_wmma.accumulator[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)] + (cast(int32, APad_global_shared_wmma.matrix_a[floordiv(vi, 16), floordiv(vk, 16), floormod(vi, 16), floormod(vk, 16)])*cast(int32, BPad_global_shared_wmma.matrix_b[floordiv(vk, 16), floordiv(vj, 16), floormod(vk, 16), floormod(vj, 16)])))
                }
              }
            }
          }
        }
      }
      for (ax0_6: int32, 0, 3200) {
        for (ax1_6: int32, 0, 128) {
          block([3200, 128], "CPad_shared_wmma.accumulator") as [v0_6, v1_6] {
            bind(v0_6, ax0_6)
            bind(v1_6, ax1_6)
            tir.reads([CPad_shared_wmma.accumulator[floordiv(v0_6, 16), floordiv(v1_6, 16), floormod(v0_6, 16), floormod(v1_6, 16)]])
            tir.writes([CPad_shared[v0_6, v1_6]])
            CPad_shared[v0_6, v1_6] = CPad_shared_wmma.accumulator[floordiv(v0_6, 16), floordiv(v1_6, 16), floormod(v0_6, 16), floormod(v1_6, 16)]
        }
      }
      for (ax0_7: int32, 0, 3200) {
        for (ax1_7: int32, 0, 128) {
          block([3200, 128], "CPad_shared") as [v0_7, v1_7] {
            where(((ax0_7 < 3136) && (ax1_7 < 64)))
            bind(v0_7, ax0_7)
            bind(v1_7, ax1_7)
            tir.reads([CPad_shared[v0_7, v1_7]])
            tir.writes([C[v0_7, v1_7]])
            C[v0_7, v1_7] = CPad_shared[v0_7, v1_7]
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