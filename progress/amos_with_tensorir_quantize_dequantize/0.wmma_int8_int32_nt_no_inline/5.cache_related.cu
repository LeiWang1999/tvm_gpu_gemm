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
    A_global = alloc_buffer(int8[16384, 16384])
    QA_shared = alloc_buffer(int8[16384, 16384])
    QA_shared_wmma.matrix_a = alloc_buffer(int8[16384, 16384])
    B_global = alloc_buffer(int8[16384, 16384])
    QB_shared = alloc_buffer(int8[16384, 16384])
    QB_shared_wmma.matrix_b = alloc_buffer(int8[16384, 16384])
    QC_wmma.accumulator = alloc_buffer(int32[16384, 16384])
     {
      for (ax0: int32, 0, 16384) {
        for (ax1: int32, 0, 16384) {
          block([16384, 16384], "B_global") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([B[v0, v1]])
            tir.writes([B_global[v0, v1]])
            B_global[v0, v1] = B[v0, v1]
        }
      }
      for (ax0_1: int32, 0, 16384) {
        for (ax1_1: int32, 0, 16384) {
          block([16384, 16384], "A_global") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([A[v0_1, v1_1]])
            tir.writes([A_global[v0_1, v1_1]])
            A_global[v0_1, v1_1] = A[v0_1, v1_1]
        }
      }
      for (i: int32, 0, 16384) {
        for (j: int32, 0, 16384) {
          block([16384, 16384], "Quantize_A") as [vi, vj] {
            bind(vi, i)
            bind(vj, j)
            tir.reads([A_global[vi, vj]])
            tir.writes([QA[vi, vj]])
            QA[vi, vj] = cast(int8, (@tir.round((cast(float32, A_global[vi, vj])*0.5f32), dtype=float32) - 0f32))
        }
      }
      for (i_1: int32, 0, 16384) {
        for (j_1: int32, 0, 16384) {
          block([16384, 16384], "Quantize_B") as [vi_1, vj_1] {
            bind(vi_1, i_1)
            bind(vj_1, j_1)
            tir.reads([B_global[vi_1, vj_1]])
            tir.writes([QB[vi_1, vj_1]])
            QB[vi_1, vj_1] = cast(int8, (@tir.round((cast(float32, B_global[vi_1, vj_1])*0.1f32), dtype=float32) - 0f32))
        }
      }
      for (ax0_2: int32, 0, 16384) {
        for (ax1_2: int32, 0, 16384) {
          block([16384, 16384], "QA_shared") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([QA[v0_2, v1_2]])
            tir.writes([QA_shared[v0_2, v1_2]])
            QA_shared[v0_2, v1_2] = QA[v0_2, v1_2]
        }
      }
      for (ax0_3: int32, 0, 16384) {
        for (ax1_3: int32, 0, 16384) {
          block([16384, 16384], "QA_shared_wmma.matrix_a") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([QA_shared[v0_3, v1_3]])
            tir.writes([QA_shared_wmma.matrix_a[v0_3, v1_3]])
            QA_shared_wmma.matrix_a[v0_3, v1_3] = QA_shared[v0_3, v1_3]
        }
      }
      for (ax0_4: int32, 0, 16384) {
        for (ax1_4: int32, 0, 16384) {
          block([16384, 16384], "QB_shared") as [v0_4, v1_4] {
            bind(v0_4, ax0_4)
            bind(v1_4, ax1_4)
            tir.reads([QB[v0_4, v1_4]])
            tir.writes([QB_shared[v0_4, v1_4]])
            QB_shared[v0_4, v1_4] = QB[v0_4, v1_4]
        }
      }
      for (ax0_5: int32, 0, 16384) {
        for (ax1_5: int32, 0, 16384) {
          block([16384, 16384], "QB_shared_wmma.matrix_b") as [v0_5, v1_5] {
            bind(v0_5, ax0_5)
            bind(v1_5, ax1_5)
            tir.reads([QB_shared[v0_5, v1_5]])
            tir.writes([QB_shared_wmma.matrix_b[v0_5, v1_5]])
            QB_shared_wmma.matrix_b[v0_5, v1_5] = QB_shared[v0_5, v1_5]
        }
      }
      for (i_2: int32, 0, 16384) {
        for (j_2: int32, 0, 16384) {
          for (k: int32, 0, 16384) {
            block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi_2, vj_2, vk] {
              bind(vi_2, i_2)
              bind(vj_2, j_2)
              bind(vk, k)
              tir.reads([QA_shared_wmma.matrix_a[vi_2, vk], QB_shared_wmma.matrix_b[vj_2, vk]])
              tir.writes([QC_wmma.accumulator[vi_2, vj_2]])
              with init() {
                QC_wmma.accumulator[vi_2, vj_2] = 0
              }
              QC_wmma.accumulator[vi_2, vj_2] = (QC_wmma.accumulator[vi_2, vj_2] + (cast(int32, QA_shared_wmma.matrix_a[vi_2, vk])*cast(int32, QB_shared_wmma.matrix_b[vj_2, vk])))
          }
        }
      }
      for (ax0_6: int32, 0, 16384) {
        for (ax1_6: int32, 0, 16384) {
          block([16384, 16384], "QC_wmma.accumulator") as [v0_6, v1_6] {
            bind(v0_6, ax0_6)
            bind(v1_6, ax1_6)
            tir.reads([QC_wmma.accumulator[v0_6, v1_6]])
            tir.writes([QC[v0_6, v1_6]])
            QC[v0_6, v1_6] = QC_wmma.accumulator[v0_6, v1_6]
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