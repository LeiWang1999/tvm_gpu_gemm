#[version = "0.0.5"]
@main = primfn(a: handle, b: handle, c: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float32), float32, [16384, 16384], []),
             B: Buffer(B_1: Pointer(global float32), float32, [16384, 16384], []),
             C: Buffer(C_1: Pointer(global float32), float32, [16384, 16384], [])}
  buffer_map = {a: A, b: B, c: C} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    A_local = alloc_buffer(float32[16384, 16384])
    A_local_shared = alloc_buffer(float32[16384, 16384])
    A_local_shared_local = alloc_buffer(float32[16384, 16384])
    B_shared = alloc_buffer(float32[16384, 16384])
    B_shared_local = alloc_buffer(float32[16384, 16384])
    C_local = alloc_buffer(float32[16384, 16384])
    for (i_0: int32, 0, 512) "thread_binding" {
      for (j_0: int32, 0, 64) "thread_binding" {
        for (i_1: int32, 0, 2) "thread_binding" {
          for (j_1: int32, 0, 2) "thread_binding" {
            for (i_2: int32, 0, 4) "thread_binding" {
              for (j_2: int32, 0, 32) "thread_binding" {
                for (k_0: int32, 0, 512) {
                  for (ax0_ax1_0_fused_1: int32, 0, 4) "thread_binding" {
                    for (ax0_ax1_0_fused_2: int32, 0, 32) "thread_binding" {
                      for (ax0_ax1_0_fused_0: int32, 0, 2) {
                        for (ax1_1: int32, 0, 4) "vectorized" {
                          block([16384, 16384], "A_local") as [v0, v1] {
                            bind(v0, ((i_0*32) + floordiv((((ax0_ax1_0_fused_0*128) + (ax0_ax1_0_fused_1*32)) + ax0_ax1_0_fused_2), 8)))
                            bind(v1, (((k_0*32) + (floormod((((ax0_ax1_0_fused_0*128) + (ax0_ax1_0_fused_1*32)) + ax0_ax1_0_fused_2), 8)*4)) + ax1_1))
                            tir.reads([A[v0, v1]])
                            tir.writes([A_local[v0, v1]])
                            A_local[v0, v1] = A[v0, v1]
                        }
                        for (ax0: int32, 0, 4) {
                          block([16384, 16384], "A_local_shared") as [v0_1, v1_1] {
                            bind(v0_1, ((i_0*32) + floordiv((((ax0_ax1_0_fused_0*128) + (ax0_ax1_0_fused_1*32)) + ax0_ax1_0_fused_2), 8)))
                            bind(v1_1, (((k_0*32) + (floormod((((ax0_ax1_0_fused_0*128) + (ax0_ax1_0_fused_1*32)) + ax0_ax1_0_fused_2), 8)*4)) + ax0))
                            tir.reads([A_local[v0_1, v1_1]])
                            tir.writes([A_local_shared[v1_1, v0_1]])
                            A_local_shared[v1_1, v0_1] = A_local[v0_1, v1_1]
                        }
                      }
                    }
                  }
                  for (ax0_1: int32, 0, 32) {
                    for (ax1: int32, 0, 256) {
                      block([16384, 16384], "B_shared") as [v0_2, v1_2] {
                        bind(v0_2, ((k_0*32) + ax0_1))
                        bind(v1_2, ((j_0*256) + ax1))
                        tir.reads([B[v0_2, v1_2]])
                        tir.writes([B_shared[v0_2, v1_2]])
                        B_shared[v0_2, v1_2] = B[v0_2, v1_2]
                    }
                  }
                  for (k_1: int32, 0, 32) {
                    for (ax0_2: int32, 0, 4) {
                      block([16384, 16384], "A_local_shared_local") as [v0_3, v1_3] {
                        bind(v0_3, ((((i_0*32) + (i_1*16)) + (i_2*4)) + ax0_2))
                        bind(v1_3, ((k_0*32) + k_1))
                        tir.reads([A_local_shared[v1_3, v0_3]])
                        tir.writes([A_local_shared_local[v0_3, v1_3]])
                        A_local_shared_local[v0_3, v1_3] = A_local_shared[v1_3, v0_3]
                    }
                    for (ax0_3: int32, 0, 4) {
                      block([16384, 16384], "B_shared_local") as [v0_4, v1_4] {
                        bind(v0_4, ((k_0*32) + k_1))
                        bind(v1_4, ((((j_0*256) + (j_1*128)) + (j_2*4)) + ax0_3))
                        tir.reads([B_shared[v0_4, v1_4]])
                        tir.writes([B_shared_local[v0_4, v1_4]])
                        B_shared_local[v0_4, v1_4] = B_shared[v0_4, v1_4]
                    }
                    for (i_3: int32, 0, 4) {
                      for (j_3: int32, 0, 4) {
                        block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi, vj, vk] {
                          bind(vi, ((((i_0*32) + (i_1*16)) + (i_2*4)) + i_3))
                          bind(vj, ((((j_0*256) + (j_1*128)) + (j_2*4)) + j_3))
                          bind(vk, ((k_0*32) + k_1))
                          tir.reads([A_local_shared_local[vi, vk], B_shared_local[vk, vj]])
                          tir.writes([C_local[vi, vj]])
                          with init() {
                            C_local[vi, vj] = 0f32
                          }
                          C_local[vi, vj] = (C_local[vi, vj] + (A_local_shared_local[vi, vk]*B_shared_local[vk, vj]))
                      }
                    }
                  }
                }
                for (ax0_4: int32, 0, 4) {
                  for (ax1_2: int32, 0, 4) {
                    block([16384, 16384], "C_local") as [v0_5, v1_5] {
                      bind(v0_5, ((((i_0*32) + (i_1*16)) + (i_2*4)) + ax0_4))
                      bind(v1_5, ((((j_0*256) + (j_1*128)) + (j_2*4)) + ax1_2))
                      tir.reads([C_local[v0_5, v1_5]])
                      tir.writes([C[v0_5, v1_5]])
                      C[v0_5, v1_5] = C_local[v0_5, v1_5]
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