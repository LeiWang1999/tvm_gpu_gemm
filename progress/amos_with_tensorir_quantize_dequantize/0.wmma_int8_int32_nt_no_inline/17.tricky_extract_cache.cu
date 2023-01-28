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
    A_global = alloc_buffer(int8[1024, 1024, 16, 16])
    QA_shared = alloc_buffer(int8[1024, 1024, 16, 16])
    QA_shared_wmma.matrix_a = alloc_buffer(int8[1024, 1024, 16, 16])
    B_global = alloc_buffer(int8[1024, 1024, 16, 16])
    QB_shared = alloc_buffer(int8[1024, 1024, 16, 16])
    QB_shared_wmma.matrix_b = alloc_buffer(int8[1024, 1024, 16, 16])
    QC_wmma.accumulator = alloc_buffer(int32[1024, 1024, 16, 16])
     {
      for (ax0: int32, 0, 16384) {
        for (ax1: int32, 0, 16384) {
          block([16384, 16384], "B_global") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([B[v0, v1]])
            tir.writes([B_global[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)]])
            B_global[floordiv(v0, 16), floordiv(v1, 16), floormod(v0, 16), floormod(v1, 16)] = B[v0, v1]
        }
      }
      for (ax0_1: int32, 0, 16384) {
        for (ax1_1: int32, 0, 16384) {
          block([16384, 16384], "A_global") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([A[v0_1, v1_1]])
            tir.writes([A_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]])
            A_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)] = A[v0_1, v1_1]
        }
      }
      for (i_0_0: int32, 0, 64) "thread_binding" {
        for (j_0_0_0: int32, 0, 8) "thread_binding" {
          for (j_0_0_1: int32, 0, 32) "thread_binding" {
            for (i_0_1: int32, 0, 4) "thread_binding" {
              for (j_0_1: int32, 0, 1) "thread_binding" {
                for (ax0_2: int32, 0, 256) {
                  for (ax1_2: int32, 0, 16384) {
                    block([16384, 16384], "Quantize_A") as [vi, vj] {
                      bind(vi, ((i_0_0*256) + ax0_2))
                      bind(vj, ax1_2)
                      tir.reads([A_global[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)]])
                      tir.writes([QA[vi, vj]])
                      QA[vi, vj] = cast(int8, (@tir.round((cast(float32, A_global[floordiv(vi, 16), floordiv(vj, 16), floormod(vi, 16), floormod(vj, 16)])*0.5f32), dtype=float32) - 0f32))
                  }
                }
                for (ax0_3: int32, 0, 64) {
                  for (ax1_3: int32, 0, 16384) {
                    block([16384, 16384], "Quantize_B") as [vi_1, vj_1] {
                      bind(vi_1, (((j_0_0_0*2048) + (j_0_0_1*64)) + ax0_3))
                      bind(vj_1, ax1_3)
                      tir.reads([B_global[floordiv(vi_1, 16), floordiv(vj_1, 16), floormod(vi_1, 16), floormod(vj_1, 16)]])
                      tir.writes([QB[vi_1, vj_1]])
                      QB[vi_1, vj_1] = cast(int8, (@tir.round((cast(float32, B_global[floordiv(vi_1, 16), floordiv(vj_1, 16), floormod(vi_1, 16), floormod(vj_1, 16)])*0.1f32), dtype=float32) - 0f32))
                  }
                }
                for (k_0_0: int32, 0, 256) {
                  for (ax0_0: int32, 0, 16) {
                    for (ax1_0: int32, 0, 4) {
                      for (ax0_1_1: int32, 0, 16) {
                        for (ax1_1_1: int32, 0, 16) {
                          block([16384, 16384], "QA_shared") as [v0_2, v1_2] {
                            bind(v0_2, (((i_0_0*256) + (ax0_0*16)) + ax0_1_1))
                            bind(v1_2, (((k_0_0*64) + (ax1_0*16)) + ax1_1_1))
                            tir.reads([QA[v0_2, v1_2]])
                            tir.writes([QA_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)]])
                            QA_shared[floordiv(v0_2, 16), floordiv(v1_2, 16), floormod(v0_2, 16), floormod(v1_2, 16)] = QA[v0_2, v1_2]
                        }
                      }
                    }
                  }
                  for (ax0_0_1: int32, 0, 4) {
                    for (ax1_0_1: int32, 0, 4) {
                      for (ax0_1_2: int32, 0, 16) {
                        for (ax1_1_2: int32, 0, 16) {
                          block([16384, 16384], "QB_shared") as [v0_3, v1_3] {
                            bind(v0_3, ((((j_0_0_0*2048) + (j_0_0_1*64)) + (ax0_0_1*16)) + ax0_1_2))
                            bind(v1_3, (((k_0_0*64) + (ax1_0_1*16)) + ax1_1_2))
                            tir.reads([QB[v0_3, v1_3]])
                            tir.writes([QB_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
                            QB_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)] = QB[v0_3, v1_3]
                        }
                      }
                    }
                  }
                  for (k_0_1: int32, 0, 4) {
                    for (ax0_0_2: int32, 0, 4) {
                      for (ax1_0_2: int32, 0, 1) {
                        for (ax0_1_3: int32, 0, 16) {
                          for (ax1_1_3: int32, 0, 16) {
                            block([16384, 16384], "QA_shared_wmma.matrix_a") as [v0_4, v1_4] {
                              bind(v0_4, ((((i_0_0*256) + (i_0_1*64)) + (ax0_0_2*16)) + ax0_1_3))
                              bind(v1_4, ((((k_0_0*64) + (k_0_1*16)) + (ax1_0_2*16)) + ax1_1_3))
                              tir.reads([QA_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
                              tir.writes([QA_shared_wmma.matrix_a[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]])
                              QA_shared_wmma.matrix_a[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)] = QA_shared[floordiv(v0_4, 16), floordiv(v1_4, 16), floormod(v0_4, 16), floormod(v1_4, 16)]
                          }
                        }
                      }
                    }
                    for (ax0_0_3: int32, 0, 4) {
                      for (ax1_0_3: int32, 0, 1) {
                        for (ax0_1_4: int32, 0, 16) {
                          for (ax1_1_4: int32, 0, 16) {
                            block([16384, 16384], "QB_shared_wmma.matrix_b") as [v0_5, v1_5] {
                              bind(v0_5, ((((j_0_0_0*2048) + (j_0_0_1*64)) + (ax0_0_3*16)) + ax0_1_4))
                              bind(v1_5, ((((k_0_0*64) + (k_0_1*16)) + (ax1_0_3*16)) + ax1_1_4))
                              tir.reads([QB_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
                              tir.writes([QB_shared_wmma.matrix_b[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]])
                              QB_shared_wmma.matrix_b[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)] = QB_shared[floordiv(v0_5, 16), floordiv(v1_5, 16), floormod(v0_5, 16), floormod(v1_5, 16)]
                          }
                        }
                      }
                    }
                    for (i_0_2: int32, 0, 4) {
                      for (j_0_2: int32, 0, 4) {
                        for (i_1: int32, 0, 16) {
                          for (j_1: int32, 0, 16) {
                            for (k_1: int32, 0, 16) {
                              block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi_2, vj_2, vk] {
                                bind(vi_2, ((((i_0_0*256) + (i_0_1*64)) + (i_0_2*16)) + i_1))
                                bind(vj_2, (((((j_0_0_0*2048) + (j_0_0_1*64)) + (j_0_1*64)) + (j_0_2*16)) + j_1))
                                bind(vk, (((k_0_0*64) + (k_0_1*16)) + k_1))
                                tir.reads([QA_shared_wmma.matrix_a[floordiv(vi_2, 16), floordiv(vk, 16), floormod(vi_2, 16), floormod(vk, 16)], QB_shared_wmma.matrix_b[floordiv(vj_2, 16), floordiv(vk, 16), floormod(vj_2, 16), floormod(vk, 16)]])
                                tir.writes([QC_wmma.accumulator[floordiv(vi_2, 16), floordiv(vj_2, 16), floormod(vi_2, 16), floormod(vj_2, 16)]])
                                with init() {
                                  QC_wmma.accumulator[floordiv(vi_2, 16), floordiv(vj_2, 16), floormod(vi_2, 16), floormod(vj_2, 16)] = 0
                                }
                                QC_wmma.accumulator[floordiv(vi_2, 16), floordiv(vj_2, 16), floormod(vi_2, 16), floormod(vj_2, 16)] = (QC_wmma.accumulator[floordiv(vi_2, 16), floordiv(vj_2, 16), floormod(vi_2, 16), floormod(vj_2, 16)] + (cast(int32, QA_shared_wmma.matrix_a[floordiv(vi_2, 16), floordiv(vk, 16), floormod(vi_2, 16), floormod(vk, 16)])*cast(int32, QB_shared_wmma.matrix_b[floordiv(vj_2, 16), floordiv(vk, 16), floormod(vj_2, 16), floormod(vk, 16)])))
                            }
                          }
                        }
                      }
                    }
                  }
                }
                for (ax0_0_4: int32, 0, 4) {
                  for (ax1_0_4: int32, 0, 4) {
                    for (ax0_1_5: int32, 0, 16) {
                      for (ax1_1_5: int32, 0, 16) {
                        block([16384, 16384], "QC_wmma.accumulator") as [v0_6, v1_6] {
                          bind(v0_6, ((((i_0_0*256) + (i_0_1*64)) + (ax0_0_4*16)) + ax0_1_5))
                          bind(v1_6, ((((j_0_0_0*2048) + (j_0_0_1*64)) + (ax1_0_4*16)) + ax1_1_5))
                          tir.reads([QC_wmma.accumulator[floordiv(v0_6, 16), floordiv(v1_6, 16), floormod(v0_6, 16), floormod(v1_6, 16)]])
                          tir.writes([QC[v0_6, v1_6]])
                          QC[v0_6, v1_6] = QC_wmma.accumulator[floordiv(v0_6, 16), floordiv(v1_6, 16), floormod(v0_6, 16), floormod(v1_6, 16)]
                      }
                    }
                  }
                }
                for (ax0_4: int32, 0, 64) {
                  for (ax1_4: int32, 0, 64) {
                    block([16384, 16384], "DeQuantize_C") as [vi_3, vj_3] {
                      bind(vi_3, (((i_0_0*256) + (i_0_1*64)) + ax0_4))
                      bind(vj_3, (((j_0_0_0*2048) + (j_0_0_1*64)) + ax1_4))
                      tir.reads([QC[vi_3, vj_3]])
                      tir.writes([C[vi_3, vj_3]])
                      C[vi_3, vj_3] = cast(int8, ((cast(float32, QC[vi_3, vj_3]) / 0.01f32) + 0f32))
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