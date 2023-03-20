#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [512, 512, 16, 16], []),
             B: Buffer(B_1: Pointer(global float16), float16, [512, 512, 16, 16], []),
             C: Buffer(C_1: Pointer(global float16), float16, [512, 512, 16, 16], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(float16[512, 512, 16, 16])
    A_shared_warp = alloc_buffer(float16[512, 512, 16, 16])
    B_shared = alloc_buffer(float16[512, 512, 16, 16])
    B_shared_warp = alloc_buffer(float16[512, 512, 16, 16])
    C_warp = alloc_buffer(float16[512, 512, 16, 16])
     {
      for (ax0: int32, 0, 512) {
        for (ax1: int32, 0, 512) {
          for (ax2: int32, 0, 16) {
            for (ax3: int32, 0, 16) {
              block([512, 512, 16, 16], "B_shared") as [v0, v1, v2, v3] {
                bind(v0, ax0)
                bind(v1, ax1)
                bind(v2, ax2)
                bind(v3, ax3)
                tir.reads([B[v0, v1, v2, v3]])
                tir.writes([B_shared[v0, v1, v2, v3]])
                B_shared[v0, v1, v2, v3] = B[v0, v1, v2, v3]
            }
          }
        }
      }
      for (ax0_1: int32, 0, 512) {
        for (ax1_1: int32, 0, 512) {
          for (ax2_1: int32, 0, 16) {
            for (ax3_1: int32, 0, 16) {
              block([512, 512, 16, 16], "A_shared") as [v0_1, v1_1, v2_1, v3_1] {
                bind(v0_1, ax0_1)
                bind(v1_1, ax1_1)
                bind(v2_1, ax2_1)
                bind(v3_1, ax3_1)
                tir.reads([A[v0_1, v1_1, v2_1, v3_1]])
                tir.writes([A_shared[v0_1, v1_1, v2_1, v3_1]])
                A_shared[v0_1, v1_1, v2_1, v3_1] = A[v0_1, v1_1, v2_1, v3_1]
            }
          }
        }
      }
      for (ax0_2: int32, 0, 512) {
        for (ax1_2: int32, 0, 512) {
          for (ax2_2: int32, 0, 16) {
            for (ax3_2: int32, 0, 16) {
              block([512, 512, 16, 16], "A_shared_warp") as [v0_2, v1_2, v2_2, v3_2] {
                bind(v0_2, ax0_2)
                bind(v1_2, ax1_2)
                bind(v2_2, ax2_2)
                bind(v3_2, ax3_2)
                tir.reads([A_shared[v0_2, v1_2, v2_2, v3_2]])
                tir.writes([A_shared_warp[v0_2, v1_2, v2_2, v3_2]])
                A_shared_warp[v0_2, v1_2, v2_2, v3_2] = A_shared[v0_2, v1_2, v2_2, v3_2]
            }
          }
        }
      }
      for (ax0_3: int32, 0, 512) {
        for (ax1_3: int32, 0, 512) {
          for (ax2_3: int32, 0, 16) {
            for (ax3_3: int32, 0, 16) {
              block([512, 512, 16, 16], "B_shared_warp") as [v0_3, v1_3, v2_3, v3_3] {
                bind(v0_3, ax0_3)
                bind(v1_3, ax1_3)
                bind(v2_3, ax2_3)
                bind(v3_3, ax3_3)
                tir.reads([B_shared[v0_3, v1_3, v2_3, v3_3]])
                tir.writes([B_shared_warp[v0_3, v1_3, v2_3, v3_3]])
                B_shared_warp[v0_3, v1_3, v2_3, v3_3] = B_shared[v0_3, v1_3, v2_3, v3_3]
            }
          }
        }
      }
      for (ii_0: int32, 0, 64) "thread_binding" {
        for (jj_0_0: int32, 0, 2) "thread_binding" {
          for (jj_0_1: int32, 0, 16) "thread_binding" {
            for (ii_1: int32, 0, 1) "thread_binding" {
              for (jj_1: int32, 0, 4) "thread_binding" {
                for (kk_0: int32, 0, 256) {
                  for (kk_1: int32, 0, 2) {
                    for (ii_2: int32, 0, 8) {
                      for (jj_2: int32, 0, 4) {
                        for (i: int32, 0, 16) {
                          for (j: int32, 0, 16) {
                            for (k: int32, 0, 16) {
                              block([512, 512, tir.reduce_axis(0, 512), 16, 16, tir.reduce_axis(0, 16)], "B") as [vii, vjj, vkk, vi, vj, vk] {
                                bind(vii, (((ii_0*8) + (ii_1*8)) + ii_2))
                                bind(vjj, ((((jj_0_0*256) + (jj_0_1*16)) + (jj_1*4)) + jj_2))
                                bind(vkk, ((kk_0*2) + kk_1))
                                bind(vi, i)
                                bind(vj, j)
                                bind(vk, k)
                                tir.reads([A_shared_warp[vii, vkk, vi, vk], B_shared_warp[vkk, vjj, vk, vj]])
                                tir.writes([C_warp[vii, vjj, vi, vj]])
                                with init() {
                                  C_warp[vii, vjj, vi, vj] = 0f32
                                }
                                C_warp[vii, vjj, vi, vj] = (C_warp[vii, vjj, vi, vj] + (A_shared_warp[vii, vkk, vi, vk]*B_shared_warp[vkk, vjj, vk, vj]))
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (ax0_4: int32, 0, 512) {
        for (ax1_4: int32, 0, 512) {
          for (ax2_4: int32, 0, 16) {
            for (ax3_4: int32, 0, 16) {
              block([512, 512, 16, 16], "C_warp") as [v0_4, v1_4, v2_4, v3_4] {
                bind(v0_4, ax0_4)
                bind(v1_4, ax1_4)
                bind(v2_4, ax2_4)
                bind(v3_4, ax3_4)
                tir.reads([C_warp[v0_4, v1_4, v2_4, v3_4]])
                tir.writes([C[v0_4, v1_4, v2_4, v3_4]])
                C[v0_4, v1_4, v2_4, v3_4] = C_warp[v0_4, v1_4, v2_4, v3_4]
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