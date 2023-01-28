@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float16), float16, [65536], []),
             B: Buffer(B_2: Pointer(float16), float16, [65536], []),
             C: Buffer(C_2: Pointer(float32), float32, [65536], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float16, [256, 256], []), B_1: B_3: Buffer(B_2, float16, [256, 256], []), C_1: C_3: Buffer(C_2, float32, [256, 256], [])} {
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 1;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator float32), float32, [2048]), storage_scope = wmma.accumulator;
  allocate(A.shared: Pointer(shared float16), float16, [16384]), storage_scope = shared;
  allocate(B.shared: Pointer(shared float16), float16, [4096]), storage_scope = shared;
  allocate(A.shared.wmma.matrix_a: Pointer(wmma.matrix_a float16), float16, [2048]), storage_scope = wmma.matrix_a;
  allocate(B.shared.wmma.matrix_b: Pointer(wmma.matrix_b float16), float16, [256]), storage_scope = wmma.matrix_b;
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 4;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
  attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 4 {
    for (ii.c.outer.init: int32, 0, 4) {
      for (jj.c.outer.init: int32, 0, 2) {
        assert((jj.c.outer.init == 0), "Argument BC.elem_offset has an unsatisfied constraint: ((((ii.c.outer.init*512) + (jj.c.outer.init*8)) % 256) == 0)")
        assert(True, "Argument BC.elem_offset has an unsatisfied constraint: ((BC_elem_offset % 256) == 0)")
        @tir.tvm_fill_fragment(C.wmma.accumulator, 32, 8, 16, (ii.c.outer.init*2), 0f32, dtype=handle)
      }
    }
    for (rk.outer.outer: int32, 0, 4) {
      for (ax0.outer.inner: int32, 0, 4) {
        attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
        for (ax0.inner.ax1.inner.fused.inner: int32, 0, 16) {
          A.shared_1: Buffer(A.shared, float16, [16384], [], scope="shared")[(((((threadIdx.y*8192) + (ax0.outer.inner*2048)) + (threadIdx.x*64)) + (threadIdx.z*16)) + ax0.inner.ax1.inner.fused.inner)] = A[((((((threadIdx.y*32768) + (ax0.outer.inner*8192)) + (threadIdx.x*256)) + (rk.outer.outer*64)) + (threadIdx.z*16)) + ax0.inner.ax1.inner.fused.inner)]
        }
      }
      for (ax0.outer.inner_1: int32, 0, 2) {
        for (ax1.outer.inner: int32, 0, 2) {
          attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
          for (ax0.inner.ax1.inner.fused.inner_1: int32, 0, 4) {
            let cse_var_1: int32 = (ax1.outer.inner*8)
            B.shared_1: Buffer(B.shared, float16, [4096], [], scope="shared")[(((((((threadIdx.y*2048) + (ax0.outer.inner_1*1024)) + (floordiv(threadIdx.x, 2)*64)) + (threadIdx.z*16)) + cse_var_1) + (floormod(threadIdx.x, 2)*4)) + ax0.inner.ax1.inner.fused.inner_1)] = B[(((((((((rk.outer.outer*16384) + (threadIdx.y*8192)) + (ax0.outer.inner_1*4096)) + (floordiv(threadIdx.x, 2)*256)) + (blockIdx.y*64)) + (threadIdx.z*16)) + cse_var_1) + (floormod(threadIdx.x, 2)*4)) + ax0.inner.ax1.inner.fused.inner_1)]
          }
        }
      }
      for (rk.outer.inner: int32, 0, 4) {
        for (ax0.outer: int32, 0, 4) {
          assert((rk.outer.inner == 0), "Argument buffer.elem_offset has an unsatisfied constraint: (((((threadIdx.y*8192) + (ax0.outer*2048)) + (rk.outer.inner*16)) % 512) == 0)")
          assert(True, "Argument buffer.elem_offset has an unsatisfied constraint: ((buffer_elem_offset % 512) == 0)")
          @tir.tvm_load_matrix_sync(A.shared.wmma.matrix_a, 32, 8, 16, ax0.outer, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), A.shared, (((threadIdx.y*8192) + (ax0.outer*2048)) + (rk.outer.inner*16)), 2048, 1, dtype=handle), 16, "row_major", dtype=handle)
        }
        for (ax1.outer: int32, 0, 2) {
          let cse_var_2: int32 = (ax1.outer*8)
          assert((((threadIdx.z*16) + cse_var_2) == 0), "Argument buffer.elem_offset has an unsatisfied constraint: (((((rk.outer.inner*1024) + (threadIdx.z*16)) + (ax1.outer*8)) % 128) == 0)")
          assert(True, "Argument buffer.elem_offset has an unsatisfied constraint: ((buffer_elem_offset % 128) == 0)")
          assert((ax1.outer == 0), "Argument buffer.elem_offset has an unsatisfied constraint: (((ax1.outer*8) % 128) == 0)")
          assert(True, "Argument buffer.elem_offset has an unsatisfied constraint: ((buffer_elem_offset % 128) == 0)")
          @tir.tvm_load_matrix_sync(B.shared.wmma.matrix_b, 32, 8, 16, 0, @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), B.shared, (((rk.outer.inner*1024) + (threadIdx.z*16)) + cse_var_2), 1024, 1, dtype=handle), 8, "row_major", dtype=handle)
        }
        for (ii.c.outer: int32, 0, 4) {
          for (jj.c.outer: int32, 0, 2) {
            let cse_var_3: int32 = (ii.c.outer*2)
            assert((jj.c.outer == 0), "Argument BB.elem_offset has an unsatisfied constraint: (((jj.c.outer*8) % 128) == 0)")
            assert(True, "Argument BB.elem_offset has an unsatisfied constraint: ((BB_elem_offset % 128) == 0)")
            assert(True, "Argument BC.elem_offset has an unsatisfied constraint: ((((ii.c.outer*512) + (jj.c.outer*8)) % 256) == 0)")
            assert(True, "Argument BC.elem_offset has an unsatisfied constraint: ((BC_elem_offset % 256) == 0)")
            @tir.tvm_mma_sync(C.wmma.accumulator, cse_var_3, A.shared.wmma.matrix_a, ii.c.outer, B.shared.wmma.matrix_b, 0, C.wmma.accumulator, cse_var_3, dtype=handle)
          }
        }
      }
    }
    for (ii.outer.inner: int32, 0, 4) {
      for (jj.outer.inner: int32, 0, 2) {
        let cse_var_4: int32 = (jj.outer.inner*8)
        assert((jj.outer.inner == 0), "Argument buffer.elem_offset has an unsatisfied constraint: ((((ii.outer.inner*512) + (jj.outer.inner*8)) % 256) == 0)")
        assert(True, "Argument buffer.elem_offset has an unsatisfied constraint: ((buffer_elem_offset % 256) == 0)")
        assert(((((blockIdx.y*64) + (threadIdx.z*16)) + cse_var_4) == 0), "Argument buffer.elem_offset has an unsatisfied constraint: (((((((threadIdx.y*32768) + (ii.outer.inner*8192)) + (blockIdx.y*64)) + (threadIdx.z*16)) + (jj.outer.inner*8)) % 256) == 0)")
        assert(True, "Argument buffer.elem_offset has an unsatisfied constraint: ((buffer_elem_offset % 256) == 0)")
        @tir.tvm_store_matrix_sync(C.wmma.accumulator, 32, 8, 16, (ii.outer.inner*2), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float32), C_2, (((((threadIdx.y*32768) + (ii.outer.inner*8192)) + (blockIdx.y*64)) + (threadIdx.z*16)) + cse_var_4), 8192, 2, dtype=handle), 8, "row_major", dtype=handle)
      }
    }
  }
}

