@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [1024, 1024, 16, 16], []), B_1: B_3: Buffer(B_2, int8, [1024, 1024, 16, 16], []), C_1: C_3: Buffer(C_2, int32, [1024, 1024, 16, 16], [])} {
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 256;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator int32), int32, [4096]), storage_scope = wmma.accumulator;
  allocate(A.shared: Pointer(shared int8), int8, [4096]), storage_scope = shared;
  allocate(B.shared: Pointer(shared int8), int8, [16384]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a int8), int8, [512]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b int8), int8, [2048]), storage_scope = wmma.matrix_b;
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 64;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
  attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2 {
    for (i.c.init: int32, 0, 2) {
      for (j.c.init: int32, 0, 8) {
        for (ii.c.init: int32, 0, 16) {
          for (jj.c.init: int32, 0, 16) {
            C.wmma.accumulator_1: Buffer(C.wmma.accumulator, int32, [4096], [], scope="wmma.accumulator")[((((i.c.init*2048) + (j.c.init*256)) + (ii.c.init*16)) + jj.c.init)] = 0
          }
        }
      }
    }
    for (k1.outer: int32, 0, 256) {
      for (ax0.ax1.fused.ax2.fused.ax3.fused.inner.inner.outer: int32, 0, 2) {
        let cse_var_1: int32 = (ax0.ax1.fused.ax2.fused.ax3.fused.inner.inner.outer*512)
        attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
        A.shared_1: Buffer(A.shared, int8, [4096], [], scope="shared")[ramp(((((threadIdx.y*2048) + (threadIdx.z*1024)) + cse_var_1) + (threadIdx.x*16)), 1, 16)] = A[ramp(((((((blockIdx.x*1048576) + (threadIdx.y*524288)) + (threadIdx.z*262144)) + (k1.outer*1024)) + cse_var_1) + (threadIdx.x*16)), 1, 16)]
      }
      for (ax0.ax1.fused.ax2.fused.ax3.fused.inner.inner.outer_1: int32, 0, 8) {
        attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
        B.shared_1: Buffer(B.shared, int8, [16384], [], scope="shared")[ramp(((((threadIdx.y*8192) + (threadIdx.z*4096)) + (ax0.ax1.fused.ax2.fused.ax3.fused.inner.inner.outer_1*512)) + (threadIdx.x*16)), 1, 16)] = B[ramp((((((((blockIdx.y*4194304) + (threadIdx.y*2097152)) + (threadIdx.z*1048576)) + (floordiv(ax0.ax1.fused.ax2.fused.ax3.fused.inner.inner.outer_1, 2)*262144)) + (k1.outer*1024)) + (floormod(ax0.ax1.fused.ax2.fused.ax3.fused.inner.inner.outer_1, 2)*512)) + (threadIdx.x*16)), 1, 16)]
      }
      for (k1.inner: int32, 0, 4) {
        for (ax0: int32, 0, 2) {
          @tir.tvm_load_matrix_sync(A.shared.wmma.matrix_a, 16, 16, 16, ax0, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), A.shared, (((threadIdx.y*2048) + (ax0*1024)) + (k1.inner*256)), 256, 1, dtype=handle), 16, "row_major", dtype=handle)
        }
        for (ax0_1: int32, 0, 8) {
          for (ax2: int32, 0, 16) {
            for (ax3: int32, 0, 16) {
              let cse_var_2: int32 = (ax2*16)
              B.shared.wmma.matrix_b_1: Buffer(B.shared.wmma.matrix_b, int8, [2048], [], scope="wmma.matrix_b")[(((ax0_1*256) + cse_var_2) + ax3)] = B.shared_1[(((((threadIdx.z*8192) + (ax0_1*1024)) + (k1.inner*256)) + cse_var_2) + ax3)]
            }
          }
        }
        for (i.c: int32, 0, 2) {
          for (j.c: int32, 0, 8) {
            for (ii.c: int32, 0, 16) {
              for (jj.c: int32, 0, 16) {
                for (k2: int32, 0, 16) {
                  let cse_var_5: int32 = (j.c*256)
                  let cse_var_4: int32 = (ii.c*16)
                  let cse_var_3: int32 = ((((i.c*2048) + cse_var_5) + cse_var_4) + jj.c)
                  C.wmma.accumulator_1[cse_var_3] = (C.wmma.accumulator_1[cse_var_3] + (cast(int32, A.shared.wmma.matrix_a_1: Buffer(A.shared.wmma.matrix_a, int8, [512], [], scope="wmma.matrix_a")[(((i.c*256) + cse_var_4) + k2)])*cast(int32, B.shared.wmma.matrix_b_1[((cse_var_5 + (jj.c*16)) + k2)])))
                }
              }
            }
          }
        }
      }
    }
    for (i.inner: int32, 0, 2) {
      for (j.inner: int32, 0, 8) {
        for (ii: int32, 0, 16) {
          for (jj: int32, 0, 16) {
            let cse_var_7: int32 = (j.inner*256)
            let cse_var_6: int32 = (ii*16)
            C[((((((((blockIdx.x*1048576) + (threadIdx.y*524288)) + (i.inner*262144)) + (blockIdx.y*4096)) + (threadIdx.z*2048)) + cse_var_7) + cse_var_6) + jj)] = C.wmma.accumulator_1[((((i.inner*2048) + cse_var_7) + cse_var_6) + jj)]
          }
        }
      }
    }
  }
}

