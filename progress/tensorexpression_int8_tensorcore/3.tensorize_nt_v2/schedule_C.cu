@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main", "from_legacy_te_schedule": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.shared: Pointer(shared int8), int8, [262144]), storage_scope = shared;
  allocate(B.shared: Pointer(shared int8), int8, [262144]), storage_scope = shared;
  allocate(C.wmma.accumulator: Pointer(wmma.accumulator int32), int32, [16]), storage_scope = wmma.accumulator {
    for (axis1: int32, 0, 1024) {
      for (axis2: int32, 0, 16) {
        for (axis3: int32, 0, 16) {
          A.shared_1: Buffer(A.shared, int8, [262144], [], scope="shared")[(((axis1*256) + (axis2*16)) + axis3)] = A[((((floordiv(blockIdx.x: int32, 4)*262144) + (axis2*16384)) + (axis1*16)) + axis3)]
        }
      }
    }
    for (axis1_1: int32, 0, 1024) {
      for (axis2_1: int32, 0, 16) {
        for (axis3_1: int32, 0, 16) {
          B.shared_1: Buffer(B.shared, int8, [262144], [], scope="shared")[(((axis1_1*256) + (axis2_1*16)) + axis3_1)] = B[((((blockIdx.y: int32*262144) + (axis2_1*16384)) + (axis1_1*16)) + axis3_1)]
        }
      }
    }
    for (axis2_2: int32, 0, 16) {
      for (axis3_2: int32, 0, 16) {
        C.wmma.accumulator_1: Buffer(C.wmma.accumulator, int32, [16], [], scope="wmma.accumulator", align=64)[(((((floordiv(((axis2_2 - (threadIdx.y: int32*2)) - (blockIdx.x*4)), 16)*16) + (floordiv(blockIdx.x, 4)*16)) + (floormod((((threadIdx.y*14) + (blockIdx.x*12)) + axis2_2), 16)*8)) + axis3_2) - (threadIdx.z: int32*8))] = 0
        for (rk: int32, 0, 16384) {
          let cse_var_2: int32 = floormod(rk, 16)
          let cse_var_1: int32 = (floordiv(rk, 16)*256)
          C.wmma.accumulator_1[(((((floordiv(((axis2_2 - (threadIdx.y*2)) - (blockIdx.x*4)), 16)*16) + (floordiv(blockIdx.x, 4)*16)) + (floormod((((threadIdx.y*14) + (blockIdx.x*12)) + axis2_2), 16)*8)) + axis3_2) - (threadIdx.z*8))] = (C.wmma.accumulator_1[(((((floordiv(((axis2_2 - (threadIdx.y*2)) - (blockIdx.x*4)), 16)*16) + (floordiv(blockIdx.x, 4)*16)) + (floormod((((threadIdx.y*14) + (blockIdx.x*12)) + axis2_2), 16)*8)) + axis3_2) - (threadIdx.z*8))] + (cast(int32, A.shared_1[((cse_var_1 + (axis2_2*16)) + cse_var_2)])*cast(int32, B.shared_1[((cse_var_1 + (axis3_2*16)) + cse_var_2)])))
        }
      }
    }
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 4096;
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 1024;
    attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
    attr [IterVar(threadIdx.z, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2;
    for (ii.inner: int32, 0, 2) {
      for (jj.inner: int32, 0, 8) {
        C[((((((blockIdx.x*65536) + (threadIdx.y*32768)) + (ii.inner*16384)) + (blockIdx.y*16)) + (threadIdx.z*8)) + jj.inner)] = C.wmma.accumulator_1[((ii.inner*8) + jj.inner)]
      }
    }
  }
}

