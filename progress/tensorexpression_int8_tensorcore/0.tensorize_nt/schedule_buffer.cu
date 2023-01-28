@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main", "from_legacy_te_schedule": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.global: Pointer(global int8), int8, [268435456]), storage_scope = global;
  allocate(B.global: Pointer(global int8), int8, [268435456]), storage_scope = global;
  allocate(C.global: Pointer(global int32), int32, [268435456]), storage_scope = global {
    attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1024;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 32;
    attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    for (axis2.axis3.fused.outer: int32, 0, 16) {
      A.global_1: Buffer(A.global, int8, [268435456], [])[ramp(((((blockIdx.x*262144) + (threadIdx.y*8192)) + (threadIdx.x*256)) + (axis2.axis3.fused.outer*16)), 1, 16)] = A[ramp(((((blockIdx.x*262144) + (axis2.axis3.fused.outer*16384)) + (threadIdx.y*512)) + (threadIdx.x*16)), 1, 16)]
    }
    attr [IterVar(blockIdx.x_1: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1024;
    attr [IterVar(threadIdx.y_1: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 32;
    attr [IterVar(threadIdx.x_1: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    for (axis2.axis3.fused.outer_1: int32, 0, 16) {
      B.global_1: Buffer(B.global, int8, [268435456], [])[ramp(((((blockIdx.x_1*262144) + (threadIdx.y_1*8192)) + (threadIdx.x_1*256)) + (axis2.axis3.fused.outer_1*16)), 1, 16)] = B[ramp(((((blockIdx.x_1*262144) + (axis2.axis3.fused.outer_1*16384)) + (threadIdx.y_1*512)) + (threadIdx.x_1*16)), 1, 16)]
    }
    attr [IterVar(blockIdx.x_2: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1024;
    attr [IterVar(threadIdx.y_2: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 32;
    attr [IterVar(threadIdx.x_2: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    for (axis2.axis3.fused.outer_2: int32, 0, 16) {
      C.global_1: Buffer(C.global, int32, [268435456], [])[ramp(((((blockIdx.x_2*262144) + (threadIdx.y_2*8192)) + (threadIdx.x_2*256)) + (axis2.axis3.fused.outer_2*16)), 1, 16)] = broadcast(0, 16)
      for (rk.outer: int32, 0, 1024) {
        for (rk.inner: int32, 0, 16) {
          let cse_var_2: int32 = (rk.outer*256)
          let cse_var_1: int32 = (axis2.axis3.fused.outer_2*16)
          C.global_1[ramp(((((blockIdx.x_2*262144) + (threadIdx.y_2*8192)) + (threadIdx.x_2*256)) + cse_var_1), 1, 16)] = (C.global_1[ramp(((((blockIdx.x_2*262144) + (threadIdx.y_2*8192)) + (threadIdx.x_2*256)) + cse_var_1), 1, 16)] + (broadcast(cast(int32, A.global_1[((((blockIdx.x_2*262144) + cse_var_2) + cse_var_1) + rk.inner)]), 16)*cast(int32x16, B.global_1[ramp(((((threadIdx.y_2*8388608) + (threadIdx.x_2*262144)) + cse_var_2) + rk.inner), 16, 16)])))
        }
      }
    }
    for (ii: int32, 0, 16384) {
      for (jj: int32, 0, 16384) {
        C[((ii*16384) + jj)] = C.global_1[((((floordiv(ii, 16)*262144) + (floordiv(jj, 16)*256)) + (floormod(ii, 16)*16)) + floormod(jj, 16))]
      }
    }
  }
}

