@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main", "from_legacy_te_schedule": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 4096;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator int32), int32, [16]), storage_scope = wmma.accumulator;
  allocate(A.shared: Pointer(shared int8), int8, [1024]), storage_scope = shared;
  allocate(B.shared: Pointer(shared int8), int8, [1024]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a int8), int8, [256]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b int8), int8, [256]), storage_scope = wmma.matrix_b;
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 1024;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
  attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2 {
    for (axis2.init: int32, 0, 16) {
      for (axis3.init: int32, 0, 16) {
        C.wmma.accumulator_1: Buffer(C.wmma.accumulator, int32, [16], [], scope="wmma.accumulator", align=64)[(((((floordiv(((axis2.init - (threadIdx.y*2)) - (blockIdx.x*4)), 16)*16) + (floordiv(blockIdx.x, 4)*16)) + (floormod((((threadIdx.y*14) + (blockIdx.x*12)) + axis2.init), 16)*8)) + axis3.init) - (threadIdx.z*8))] = 0
      }
    }
    for (rk.outer.outer: int32, 0, 256) {
      attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
      if @tir.likely(((floordiv((floordiv(threadIdx.x, 16) + threadIdx.z), 2) + threadIdx.y) < 2), dtype=bool) {
        for (axis0.axis1.fused.axis2.fused.axis3.fused.inner.inner.inner.inner.s: int32, 0, 16) {
          if @tir.likely((((((threadIdx.y*512) + (threadIdx.z*256)) + (threadIdx.x*16)) + axis0.axis1.fused.axis2.fused.axis3.fused.inner.inner.inner.inner.s) < 1024), dtype=bool) {
            if @tir.likely(((floordiv(threadIdx.x, 16) + threadIdx.z) < 2), dtype=bool) {
              if @tir.likely((threadIdx.x < 16), dtype=bool) {
                A.shared_1: Buffer(A.shared, int8, [1024], [], scope="shared")[((((threadIdx.y*512) + (threadIdx.z*256)) + (threadIdx.x*16)) + axis0.axis1.fused.axis2.fused.axis3.fused.inner.inner.inner.inner.s)] = A[((((((floordiv(blockIdx.x, 4)*262144) + (threadIdx.x*16384)) + (rk.outer.outer*64)) + (threadIdx.y*32)) + (threadIdx.z*16)) + axis0.axis1.fused.axis2.fused.axis3.fused.inner.inner.inner.inner.s)]
              }
            }
          }
        }
      }
      attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
      if @tir.likely(((floordiv((floordiv(threadIdx.x, 16) + threadIdx.z), 2) + threadIdx.y) < 2), dtype=bool) {
        for (axis0.axis1.fused.axis2.fused.axis3.fused.inner.inner.inner.inner.s_1: int32, 0, 16) {
          if @tir.likely((((((threadIdx.y*512) + (threadIdx.z*256)) + (threadIdx.x*16)) + axis0.axis1.fused.axis2.fused.axis3.fused.inner.inner.inner.inner.s_1) < 1024), dtype=bool) {
            if @tir.likely(((floordiv(threadIdx.x, 16) + threadIdx.z) < 2), dtype=bool) {
              if @tir.likely((threadIdx.x < 16), dtype=bool) {
                B.shared_1: Buffer(B.shared, int8, [1024], [], scope="shared")[((((threadIdx.y*512) + (threadIdx.z*256)) + (threadIdx.x*16)) + axis0.axis1.fused.axis2.fused.axis3.fused.inner.inner.inner.inner.s_1)] = B[((((((blockIdx.y*262144) + (threadIdx.x*16384)) + (rk.outer.outer*64)) + (threadIdx.y*32)) + (threadIdx.z*16)) + axis0.axis1.fused.axis2.fused.axis3.fused.inner.inner.inner.inner.s_1)]
              }
            }
          }
        }
      }
      for (rk.outer.inner: int32, 0, 4) {
        for (ax0: int32, 0, 16) {
          for (ax1: int32, 0, 16) {
            let cse_var_1: int32 = (ax0*16)
            A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, int8, [256], [], scope="wmma.matrix_a")[(cse_var_1 + ax1)] = A.shared_1[(((rk.outer.inner*256) + cse_var_1) + ax1)]
          }
        }
        for (ax0_1: int32, 0, 16) {
          for (ax1_1: int32, 0, 16) {
            let cse_var_2: int32 = (ax0_1*16)
            B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, int8, [256], [], scope="wmma.matrix_b")[(cse_var_2 + ax1_1)] = B.shared_1[(((rk.outer.inner*256) + cse_var_2) + ax1_1)]
          }
        }
        for (axis2: int32, 0, 16) {
          for (axis3: int32, 0, 16) {
            for (rk.inner: int32, 0, 16) {
              C.wmma.accumulator_1[(((((floordiv(((axis2 - (threadIdx.y*2)) - (blockIdx.x*4)), 16)*16) + (floordiv(blockIdx.x, 4)*16)) + (floormod((((threadIdx.y*14) + (blockIdx.x*12)) + axis2), 16)*8)) + axis3) - (threadIdx.z*8))] = (C.wmma.accumulator_1[(((((floordiv(((axis2 - (threadIdx.y*2)) - (blockIdx.x*4)), 16)*16) + (floordiv(blockIdx.x, 4)*16)) + (floormod((((threadIdx.y*14) + (blockIdx.x*12)) + axis2), 16)*8)) + axis3) - (threadIdx.z*8))] + (cast(int32, A.shared.wmma.matrix_a_1[((axis2*16) + rk.inner)])*cast(int32, B.shared.wmma.matrix_b_1[((axis3*16) + rk.inner)])))
            }
          }
        }
      }
    }
    for (ii.inner: int32, 0, 2) {
      for (jj.inner: int32, 0, 8) {
        C[((((((blockIdx.x*65536) + (threadIdx.y*32768)) + (ii.inner*16384)) + (blockIdx.y*16)) + (threadIdx.z*8)) + jj.inner)] = C.wmma.accumulator_1[((ii.inner*8) + jj.inner)]
      }
    }
  }
}

