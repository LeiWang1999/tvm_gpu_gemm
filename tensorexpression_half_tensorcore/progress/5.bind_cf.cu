@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [65536], []),
             B: Buffer(B_2: Pointer(float16), float16, [65536], []),
             C: Buffer(C_2: Pointer(float32), float32, [65536], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [256, 256], []), B_1: B_3: Buffer(B_2, float16, [256, 256], []), C_1: C_3: Buffer(C_2, float32, [256, 256], [])} {
  allocate(A.shared: Pointer(shared float16), float16, [16384]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [16384]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [8192]), storage_scope = wmma.matrix_b {
    for (ax0: int32, 0, 64) {
      for (ax1: int32, 0, 256) {
        let cse_var_1: int32 = (ax0*256)
        A.shared_1: Buffer(A.shared, float16, [16384], [], scope="shared")[(cse_var_1 + ax1)] = A[((((blockIdx.y: int32*32768) + (threadIdx.z: int32*16384)) + cse_var_1) + ax1)]
      }
    }
    for (ax0_1: int32, 0, 64) {
      for (ax1_1: int32, 0, 256) {
        let cse_var_2: int32 = ((ax0_1*256) + ax1_1)
        A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, float16, [16384], [], scope="wmma.matrix_a")[cse_var_2] = A.shared_1[cse_var_2]
      }
    }
    for (ax0_2: int32, 0, 32) {
      for (ax1_2: int32, 0, 256) {
        let cse_var_3: int32 = (ax0_2*256)
        A.shared_2: Buffer(A.shared, float16, [8192], [], scope="shared")[(cse_var_3 + ax1_2)] = B[((((blockIdx.x: int32*16384) + (threadIdx.y: int32*8192)) + cse_var_3) + ax1_2)]
      }
    }
    for (ax0_3: int32, 0, 32) {
      for (ax1_3: int32, 0, 256) {
        let cse_var_4: int32 = ((ax0_3*256) + ax1_3)
        B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, float16, [8192], [], scope="wmma.matrix_b")[cse_var_4] = A.shared_2[cse_var_4]
      }
    }
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 2;
    allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [2048]), storage_scope = wmma.accumulator;
    allocate(C.wmma.accumulator.shared: Pointer(shared float32), float32, [8192]), storage_scope = shared;
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 4 {
      attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2;
      attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2 {
        for (ii.c: int32, 0, 64) {
          for (jj.c: int32, 0, 32) {
            C.wmma.accumulator_1: Buffer(C.wmma.accumulator, float32, [2048], [], scope="wmma.accumulator")[((ii.c*32) + jj.c)] = 0f32
            for (rk: int32, 0, 256) {
              let cse_var_5: int32 = ((ii.c*32) + jj.c)
              C.wmma.accumulator_1[cse_var_5] = (C.wmma.accumulator_1[cse_var_5] + (cast(float32, A.shared.wmma.matrix_a_1[((ii.c*256) + rk)])*cast(float32, B.shared.wmma.matrix_b_1[((jj.c*256) + rk)])))
            }
          }
        }
        for (ax0.outer.inner: int32, 0, 2) {
          for (ax1.outer.inner: int32, 0, 4) {
            for (ax0.inner: int32, 0, 32) {
              for (ax1.inner: int32, 0, 8) {
                let cse_var_6: int32 = (ax1.outer.inner*8)
                C.wmma.accumulator.shared_1: Buffer(C.wmma.accumulator.shared, float32, [8192], [], scope="shared")[((((((threadIdx.z*4096) + (ax0.outer.inner*2048)) + (ax0.inner*64)) + (threadIdx.y*32)) + cse_var_6) + ax1.inner)] = C.wmma.accumulator_1[((((ax0.outer.inner*1024) + (ax0.inner*32)) + cse_var_6) + ax1.inner)]
              }
            }
          }
        }
      }
      for (ii.inner.jj.inner.fused.outer.outer.outer: int32, 0, 64) {
        attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2;
        attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
        attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
        C[((((((blockIdx.y*32768) + (ii.inner.jj.inner.fused.outer.outer.outer*512)) + (threadIdx.z*256)) + (blockIdx.x*64)) + (threadIdx.y*32)) + threadIdx.x)] = C.wmma.accumulator.shared_1[((((ii.inner.jj.inner.fused.outer.outer.outer*128) + (threadIdx.z*64)) + (threadIdx.y*32)) + threadIdx.x)]
      }
    }
  }
}

