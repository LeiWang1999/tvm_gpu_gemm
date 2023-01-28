#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {AT: Buffer(AT_1: Pointer(global float32), float32, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global float32), float32, [16384, 16384], []),
             C: Buffer(C_1: Pointer(global float32), float32, [16384, 16384], [])}
  buffer_map = {a: AT, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    AT_shared = alloc_buffer(float32[16384, 16384])
    AT_shared_local = alloc_buffer(float32[16384, 16384])
    B_shared = alloc_buffer(float32[16384, 16384])
    B_shared_local = alloc_buffer(float32[16384, 16384])
    C_local = alloc_buffer(float32[16384, 16384])
    for (i_0: int32, 0, 128) "thread_binding" {
      for (j_0: int32, 0, 128) "thread_binding" {
        for (i_1: int32, 0, 2) "thread_binding" {
          for (j_1: int32, 0, 2) "thread_binding" {
            for (i_2: int32, 0, 16) "thread_binding" {
              for (j_2: int32, 0, 16) "thread_binding" {
                for (i_3_init: int32, 0, 4) {
                  for (j_3_init: int32, 0, 4) {
                    block([16384, 16384], "B_init") as [vi, vj] {
                      bind(vi, ((((i_0*128) + (i_1*64)) + (i_2*4)) + i_3_init))
                      bind(vj, ((((j_0*128) + (j_1*64)) + (j_2*4)) + j_3_init))
                      tir.reads([])
                      tir.writes([C_local[vi, vj]])
                      C_local[vi, vj] = 0f32
                  }
                }
                for (k_0: int32, 0, 1024) {
                  for (ax0_ax1_fused_0: int32, 0, 2) {
                    for (ax0_ax1_fused_1: int32, 0, 16) "thread_binding" {
                      for (ax0_ax1_fused_2: int32, 0, 16) "thread_binding" {
                        for (ax0_ax1_fused_3: int32, 0, 4) "vectorized" {
                          block([16384, 16384], "AT_shared") as [v0, v1] {
                            bind(v0, ((k_0*16) + floordiv(((((ax0_ax1_fused_0*1024) + (ax0_ax1_fused_1*64)) + (ax0_ax1_fused_2*4)) + ax0_ax1_fused_3), 128)))
                            bind(v1, ((i_0*128) + floormod(((((ax0_ax1_fused_0*1024) + (ax0_ax1_fused_1*64)) + (ax0_ax1_fused_2*4)) + ax0_ax1_fused_3), 128)))
                            tir.reads([AT[v0, v1]])
                            tir.writes([AT_shared[v0, v1]])
                            AT_shared[v0, v1] = AT[v0, v1]
                        }
                      }
                    }
                  }
                  for (ax0_ax1_fused_0_1: int32, 0, 2) {
                    for (ax0_ax1_fused_1_1: int32, 0, 16) "thread_binding" {
                      for (ax0_ax1_fused_2_1: int32, 0, 16) "thread_binding" {
                        for (ax0_ax1_fused_3_1: int32, 0, 4) "vectorized" {
                          block([16384, 16384], "B_shared") as [v0_1, v1_1] {
                            bind(v0_1, ((k_0*16) + floordiv(((((ax0_ax1_fused_0_1*1024) + (ax0_ax1_fused_1_1*64)) + (ax0_ax1_fused_2_1*4)) + ax0_ax1_fused_3_1), 128)))
                            bind(v1_1, ((j_0*128) + floormod(((((ax0_ax1_fused_0_1*1024) + (ax0_ax1_fused_1_1*64)) + (ax0_ax1_fused_2_1*4)) + ax0_ax1_fused_3_1), 128)))
                            tir.reads([B[v0_1, v1_1]])
                            tir.writes([B_shared[v0_1, v1_1]])
                            B_shared[v0_1, v1_1] = B[v0_1, v1_1]
                        }
                      }
                    }
                  }
                  for (k_1: int32, 0, 16) {
                    for (ax0: int32, 0, 4) {
                      block([16384, 16384], "AT_shared_local") as [v0_2, v1_2] {
                        bind(v0_2, ((k_0*16) + k_1))
                        bind(v1_2, ((((i_0*128) + (i_1*64)) + (i_2*4)) + ax0))
                        tir.reads([AT_shared[v0_2, v1_2]])
                        tir.writes([AT_shared_local[v0_2, v1_2]])
                        AT_shared_local[v0_2, v1_2] = AT_shared[v0_2, v1_2]
                    }
                    for (ax0_1: int32, 0, 4) {
                      block([16384, 16384], "B_shared_local") as [v0_3, v1_3] {
                        bind(v0_3, ((k_0*16) + k_1))
                        bind(v1_3, ((((j_0*128) + (j_1*64)) + (j_2*4)) + ax0_1))
                        tir.reads([B_shared[v0_3, v1_3]])
                        tir.writes([B_shared_local[v0_3, v1_3]])
                        B_shared_local[v0_3, v1_3] = B_shared[v0_3, v1_3]
                    }
                    for (i_3: int32, 0, 4) {
                      for (j_3: int32, 0, 4) {
                        block([16384, 16384, tir.reduce_axis(0, 16384)], "B_update") as [vi_1, vj_1, vk] {
                          bind(vi_1, ((((i_0*128) + (i_1*64)) + (i_2*4)) + i_3))
                          bind(vj_1, ((((j_0*128) + (j_1*64)) + (j_2*4)) + j_3))
                          bind(vk, ((k_0*16) + k_1))
                          tir.reads([C_local[vi_1, vj_1], AT_shared_local[vk, vi_1], B_shared_local[vk, vj_1]])
                          tir.writes([C_local[vi_1, vj_1]])
                          C_local[vi_1, vj_1] = (C_local[vi_1, vj_1] + (AT_shared_local[vk, vi_1]*B_shared_local[vk, vj_1]))
                      }
                    }
                  }
                }
                for (ax0_2: int32, 0, 4) {
                  for (ax1: int32, 0, 4) {
                    block([16384, 16384], "C_local") as [v0_4, v1_4] {
                      bind(v0_4, ((((i_0*128) + (i_1*64)) + (i_2*4)) + ax0_2))
                      bind(v1_4, ((((j_0*128) + (j_1*64)) + (j_2*4)) + ax1))
                      tir.reads([C_local[v0_4, v1_4]])
                      tir.writes([C[v0_4, v1_4]])
                      C[v0_4, v1_4] = C_local[v0_4, v1_4]
                  }
                }
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