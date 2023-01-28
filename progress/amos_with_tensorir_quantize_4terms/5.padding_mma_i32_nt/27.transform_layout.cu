#[version = "0.0.5"]
@main = primfn(A_handle: handle, B_handle: handle, C_handle: handle, PA_handle: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_1: Pointer(global int8), int8, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global int8), int8, [16384, 16384], []),
             C: Buffer(C_1: Pointer(global int32), int32, [16384, 16384], []),
             PA: Buffer(PA_1: Pointer(global int32), int32, [16384], [])}
  buffer_map = {A_handle: A, B_handle: B, C_handle: C, PA_handle: PA} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_shared = alloc_buffer(int8[16384, 16384])
    B_shared = alloc_buffer(int8[16384, 16384])
    A_shared_warp = alloc_buffer(int8[1024, 512, 32, 16])
    B_shared_warp = alloc_buffer(int8[1024, 512, 32, 16])
    C_warp = alloc_buffer(int32[1024, 1024, 32, 8])
    for (i_0: int32, 0, 128) "thread_binding" {
      for (j_0: int32, 0, 64) "thread_binding" {
        for (i_1_0: int32, 0, 2) "thread_binding" {
          for (j_1_0: int32, 0, 4) "thread_binding" {
            for (i_1_1_0_init: int32, 0, 4) {
              for (j_1_1_0_init: int32, 0, 4) {
                for (i_1_1_1_init: int32, 0, 16) {
                  for (j_1_1_1_init: int32, 0, 16) {
                    block([16384, 16384], "B_init") as [vi, vj] {
                      bind(vi, ((((i_0*128) + (i_1_0*64)) + (i_1_1_0_init*16)) + i_1_1_1_init))
                      bind(vj, ((((j_0*256) + (j_1_0*64)) + (j_1_1_0_init*16)) + j_1_1_1_init))
                      tir.reads([])
                      tir.writes([C_warp[floordiv(vi, 16), floordiv(vj, 16), ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(floormod(vj, 16), 8)*4) + (floordiv(floormod(vi, 16), 8)*2)) + floormod(vj, 2))]])
                      C_warp[floordiv(vi, 16), floordiv(vj, 16), ((floormod(vi, 8)*4) + floordiv(floormod(vj, 8), 2)), (((floordiv(floormod(vj, 16), 8)*4) + (floordiv(floormod(vi, 16), 8)*2)) + floormod(vj, 2))] = 0
                  }
                }
              }
            }
            for (k_0: int32, 0, 256) {
              for (ax0_ax1_fused_0: int32, 0, 4) {
                for (ax0_ax1_fused_1: int32, 0, 4) "thread_binding" {
                  for (ax0_ax1_fused_2: int32, 0, 32) "thread_binding" {
                    for (ax0_ax1_fused_3: int32, 0, 16) "vectorized" {
                      block([16384, 16384], "A_shared") as [v0, v1] {
                        bind(v0, ((i_0*128) + floordiv(((((ax0_ax1_fused_0*2048) + (ax0_ax1_fused_1*512)) + (ax0_ax1_fused_2*16)) + ax0_ax1_fused_3), 64)))
                        bind(v1, ((k_0*64) + floormod(((((ax0_ax1_fused_0*2048) + (ax0_ax1_fused_1*512)) + (ax0_ax1_fused_2*16)) + ax0_ax1_fused_3), 64)))
                        tir.reads([A[v0, v1]])
                        tir.writes([A_shared[v0, v1]])
                        tir.attrs({"buffer_dim_align": [[0, 0, 32, 0]]})
                        A_shared[v0, v1] = A[v0, v1]
                    }
                  }
                }
              }
              for (ax0_ax1_fused_0_1: int32, 0, 8) {
                for (ax0_ax1_fused_1_1: int32, 0, 4) "thread_binding" {
                  for (ax0_ax1_fused_2_1: int32, 0, 32) "thread_binding" {
                    for (ax0_ax1_fused_3_1: int32, 0, 16) "vectorized" {
                      block([16384, 16384], "B_shared") as [v0_1, v1_1] {
                        bind(v0_1, ((j_0*256) + floordiv(((((ax0_ax1_fused_0_1*2048) + (ax0_ax1_fused_1_1*512)) + (ax0_ax1_fused_2_1*16)) + ax0_ax1_fused_3_1), 64)))
                        bind(v1_1, ((k_0*64) + floormod(((((ax0_ax1_fused_0_1*2048) + (ax0_ax1_fused_1_1*512)) + (ax0_ax1_fused_2_1*16)) + ax0_ax1_fused_3_1), 64)))
                        tir.reads([B[v0_1, v1_1]])
                        tir.writes([B_shared[v0_1, v1_1]])
                        tir.attrs({"buffer_dim_align": [[0, 0, 32, 0]]})
                        B_shared[v0_1, v1_1] = B[v0_1, v1_1]
                    }
                  }
                }
              }
              for (i_1_1_0: int32, 0, 4) {
                for (j_1_1_0: int32, 0, 4) {
                  for (k_1_0: int32, 0, 2) {
                    for (ax0: int32, 0, 16) {
                      for (ax1: int32, 0, 32) {
                        block([16384, 16384], "A_shared_warp") as [v0_2, v1_2] {
                          bind(v0_2, ((((i_0*128) + (i_1_0*64)) + (i_1_1_0*16)) + ax0))
                          bind(v1_2, (((k_0*64) + (k_1_0*32)) + ax1))
                          tir.reads([A_shared[v0_2, v1_2]])
                          tir.writes([A_shared_warp[floordiv(v0_2, 16), floordiv(v1_2, 32), ((floormod(v0_2, 8)*4) + floordiv(floormod(v1_2, 16), 4)), (((floordiv(floormod(v1_2, 32), 16)*8) + (floordiv(floormod(v0_2, 16), 8)*4)) + floormod(v1_2, 4))]])
                          A_shared_warp[floordiv(v0_2, 16), floordiv(v1_2, 32), ((floormod(v0_2, 8)*4) + floordiv(floormod(v1_2, 16), 4)), (((floordiv(floormod(v1_2, 32), 16)*8) + (floordiv(floormod(v0_2, 16), 8)*4)) + floormod(v1_2, 4))] = A_shared[v0_2, v1_2]
                      }
                    }
                    for (ax0_1: int32, 0, 16) {
                      for (ax1_1: int32, 0, 32) {
                        block([16384, 16384], "B_shared_warp") as [v0_3, v1_3] {
                          bind(v0_3, ((((j_0*256) + (j_1_0*64)) + (j_1_1_0*16)) + ax0_1))
                          bind(v1_3, (((k_0*64) + (k_1_0*32)) + ax1_1))
                          tir.reads([B_shared[v0_3, v1_3]])
                          tir.writes([B_shared_warp[floordiv(v0_3, 16), floordiv(v1_3, 32), ((floormod(v0_3, 8)*4) + floordiv(floormod(v1_3, 16), 4)), (((floordiv(floormod(v1_3, 32), 16)*8) + (floordiv(floormod(v0_3, 16), 8)*4)) + floormod(v1_3, 4))]])
                          B_shared_warp[floordiv(v0_3, 16), floordiv(v1_3, 32), ((floormod(v0_3, 8)*4) + floordiv(floormod(v1_3, 16), 4)), (((floordiv(floormod(v1_3, 32), 16)*8) + (floordiv(floormod(v0_3, 16), 8)*4)) + floormod(v1_3, 4))] = B_shared[v0_3, v1_3]
                      }
                    }
                    for (i_1_1_1: int32, 0, 16) {
                      for (j_1_1_1: int32, 0, 16) {
                        for (k_1_1: int32, 0, 32) {
                          block([16384, 16384, tir.reduce_axis(0, 16384)], "B_update") as [vi_1, vj_1, vk] {
                            bind(vi_1, ((((i_0*128) + (i_1_0*64)) + (i_1_1_0*16)) + i_1_1_1))
                            bind(vj_1, ((((j_0*256) + (j_1_0*64)) + (j_1_1_0*16)) + j_1_1_1))
                            bind(vk, (((k_0*64) + (k_1_0*32)) + k_1_1))
                            tir.reads([C_warp[floordiv(vi_1, 16), floordiv(vj_1, 16), ((floormod(vi_1, 8)*4) + floordiv(floormod(vj_1, 8), 2)), (((floordiv(floormod(vj_1, 16), 8)*4) + (floordiv(floormod(vi_1, 16), 8)*2)) + floormod(vj_1, 2))], A_shared_warp[floordiv(vi_1, 16), floordiv(vk, 32), ((floormod(vi_1, 8)*4) + floordiv(floormod(vk, 16), 4)), (((floordiv(floormod(vk, 32), 16)*8) + (floordiv(floormod(vi_1, 16), 8)*4)) + floormod(vk, 4))], B_shared_warp[floordiv(vj_1, 16), floordiv(vk, 32), ((floormod(vj_1, 8)*4) + floordiv(floormod(vk, 16), 4)), (((floordiv(floormod(vk, 32), 16)*8) + (floordiv(floormod(vj_1, 16), 8)*4)) + floormod(vk, 4))]])
                            tir.writes([C_warp[floordiv(vi_1, 16), floordiv(vj_1, 16), ((floormod(vi_1, 8)*4) + floordiv(floormod(vj_1, 8), 2)), (((floordiv(floormod(vj_1, 16), 8)*4) + (floordiv(floormod(vi_1, 16), 8)*2)) + floormod(vj_1, 2))]])
                            C_warp[floordiv(vi_1, 16), floordiv(vj_1, 16), ((floormod(vi_1, 8)*4) + floordiv(floormod(vj_1, 8), 2)), (((floordiv(floormod(vj_1, 16), 8)*4) + (floordiv(floormod(vi_1, 16), 8)*2)) + floormod(vj_1, 2))] = (C_warp[floordiv(vi_1, 16), floordiv(vj_1, 16), ((floormod(vi_1, 8)*4) + floordiv(floormod(vj_1, 8), 2)), (((floordiv(floormod(vj_1, 16), 8)*4) + (floordiv(floormod(vi_1, 16), 8)*2)) + floormod(vj_1, 2))] + (cast(int32, A_shared_warp[floordiv(vi_1, 16), floordiv(vk, 32), ((floormod(vi_1, 8)*4) + floordiv(floormod(vk, 16), 4)), (((floordiv(floormod(vk, 32), 16)*8) + (floordiv(floormod(vi_1, 16), 8)*4)) + floormod(vk, 4))])*cast(int32, B_shared_warp[floordiv(vj_1, 16), floordiv(vk, 32), ((floormod(vj_1, 8)*4) + floordiv(floormod(vk, 16), 4)), (((floordiv(floormod(vk, 32), 16)*8) + (floordiv(floormod(vj_1, 16), 8)*4)) + floormod(vk, 4))])))
                          block([16384, tir.reduce_axis(0, 16384)], "Pre_compute_A") as [vi_2, vk_1] {
                            bind(vi_2, ((((i_0*128) + (i_1_0*64)) + (i_1_1_0*16)) + i_1_1_1))
                            bind(vk_1, (((k_0*64) + (k_1_0*32)) + k_1_1))
                            tir.reads([A_shared[vi_2, vk_1]])
                            tir.writes([PA[vi_2]])
                            with init() {
                              PA[vi_2] = 0
                            }
                            PA[vi_2] = (PA[vi_2] + (1*cast(int32, A_shared[vi_2, vk_1])))
                        }
                      }
                    }
                  }
                }
              }
            }
            for (ax0_0: int32, 0, 4) {
              for (ax1_0: int32, 0, 4) {
                for (ax0_1_1: int32, 0, 16) {
                  for (ax1_1_1: int32, 0, 16) {
                    block([16384, 16384], "C_warp") as [v0_4, v1_4] {
                      bind(v0_4, ((((i_0*128) + (i_1_0*64)) + (ax0_0*16)) + ax0_1_1))
                      bind(v1_4, ((((j_0*256) + (j_1_0*64)) + (ax1_0*16)) + ax1_1_1))
                      tir.reads([C_warp[floordiv(v0_4, 16), floordiv(v1_4, 16), ((floormod(v0_4, 8)*4) + floordiv(floormod(v1_4, 8), 2)), (((floordiv(floormod(v1_4, 16), 8)*4) + (floordiv(floormod(v0_4, 16), 8)*2)) + floormod(v1_4, 2))]])
                      tir.writes([C[v0_4, v1_4]])
                      C[v0_4, v1_4] = C_warp[floordiv(v0_4, 16), floordiv(v1_4, 16), ((floormod(v0_4, 8)*4) + floordiv(floormod(v1_4, 8), 2)), (((floordiv(floormod(v1_4, 16), 8)*4) + (floordiv(floormod(v0_4, 16), 8)*2)) + floormod(v1_4, 2))]
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