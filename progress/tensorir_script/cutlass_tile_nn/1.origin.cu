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
     {
      for (ax0: int32, 0, 16384) {
        for (ax1: int32, 0, 16384) {
          block([16384, 16384], "B_shared") as [v0, v1] {
            bind(v0, ax0)
            bind(v1, ax1)
            tir.reads([B[v0, v1]])
            tir.writes([B_shared[v0, v1]])
            B_shared[v0, v1] = B[v0, v1]
        }
      }
      for (ax0_1: int32, 0, 16384) {
        for (ax1_1: int32, 0, 16384) {
          block([16384, 16384], "A_local") as [v0_1, v1_1] {
            bind(v0_1, ax0_1)
            bind(v1_1, ax1_1)
            tir.reads([A[v0_1, v1_1]])
            tir.writes([A_local[v0_1, v1_1]])
            A_local[v0_1, v1_1] = A[v0_1, v1_1]
        }
      }
      for (ax0_2: int32, 0, 16384) {
        for (ax1_2: int32, 0, 16384) {
          block([16384, 16384], "A_local_shared") as [v0_2, v1_2] {
            bind(v0_2, ax0_2)
            bind(v1_2, ax1_2)
            tir.reads([A_local[v0_2, v1_2]])
            tir.writes([A_local_shared[v0_2, v1_2]])
            A_local_shared[v0_2, v1_2] = A_local[v0_2, v1_2]
        }
      }
      for (ax0_3: int32, 0, 16384) {
        for (ax1_3: int32, 0, 16384) {
          block([16384, 16384], "A_local_shared_local") as [v0_3, v1_3] {
            bind(v0_3, ax0_3)
            bind(v1_3, ax1_3)
            tir.reads([A_local_shared[v0_3, v1_3]])
            tir.writes([A_local_shared_local[v0_3, v1_3]])
            A_local_shared_local[v0_3, v1_3] = A_local_shared[v0_3, v1_3]
        }
      }
      for (ax0_4: int32, 0, 16384) {
        for (ax1_4: int32, 0, 16384) {
          block([16384, 16384], "B_shared_local") as [v0_4, v1_4] {
            bind(v0_4, ax0_4)
            bind(v1_4, ax1_4)
            tir.reads([B_shared[v0_4, v1_4]])
            tir.writes([B_shared_local[v0_4, v1_4]])
            B_shared_local[v0_4, v1_4] = B_shared[v0_4, v1_4]
        }
      }
      for (i: int32, 0, 16384) {
        for (j: int32, 0, 16384) {
          for (k: int32, 0, 16384) {
            block([16384, 16384, tir.reduce_axis(0, 16384)], "B") as [vi, vj, vk] {
              bind(vi, i)
              bind(vj, j)
              bind(vk, k)
              tir.reads([A_local_shared_local[vi, vk], B_shared_local[vk, vj]])
              tir.writes([C_local[vi, vj]])
              with init() {
                C_local[vi, vj] = 0f32
              }
              C_local[vi, vj] = (C_local[vi, vj] + (A_local_shared_local[vi, vk]*B_shared_local[vk, vj]))
          }
        }
      }
      for (ax0_5: int32, 0, 16384) {
        for (ax1_5: int32, 0, 16384) {
          block([16384, 16384], "C_local") as [v0_5, v1_5] {
            bind(v0_5, ax0_5)
            bind(v1_5, ax1_5)
            tir.reads([C_local[v0_5, v1_5]])
            tir.writes([C[v0_5, v1_5]])
            C[v0_5, v1_5] = C_local[v0_5, v1_5]
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