@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main", "from_legacy_te_schedule": True}
  buffers = {A: Buffer(A_2: Pointer(int8), int8, [268435456], []),
             B: Buffer(B_2: Pointer(int8), int8, [268435456], []),
             C: Buffer(C_2: Pointer(int32), int32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, int8, [16384, 16384], []), B_1: B_3: Buffer(B_2, int8, [16384, 16384], []), C_1: C_3: Buffer(C_2, int32, [16384, 16384], [])} {
  allocate(A.global: Pointer(global int8), int8, [268435456]), storage_scope = global;
  allocate(B.global: Pointer(global int8), int8, [4194304]), storage_scope = global;
  allocate(B.global.shared: Pointer(shared int8), int8, [4194304]), storage_scope = shared;
  allocate(C.wmma.accumulator.global: Pointer(global int32), int32, [268435456]), storage_scope = global {
    for (axis0: int32, 0, 1024) {
      for (axis1: int32, 0, 1024) {
        for (axis2: int32, 0, 16) {
          for (axis3: int32, 0, 16) {
            let cse_var_1: int32 = (axis0*262144)
            A.global_1: Buffer(A.global, int8, [268435456], [])[(((cse_var_1 + (axis1*256)) + (axis2*16)) + axis3)] = A[(((cse_var_1 + (axis2*16384)) + (axis1*16)) + axis3)]
          }
        }
      }
    }
    if @tir.likely((blockIdx.y: int32 < 64), dtype=bool) {
      for (axis0.idx: int32, 0, 16) {
        for (axis1_1: int32, 0, 1024) {
          for (axis2_1: int32, 0, 16) {
            if @tir.likely((0 <= (((blockIdx.y*256) + (axis0.idx*16)) + axis2_1)), dtype=bool) {
              for (axis3_1: int32, 0, 16) {
                let cse_var_2: int32 = (axis0.idx*262144)
                B.global_1: Buffer(B.global, int8, [4194304], [])[(((cse_var_2 + (axis1_1*256)) + (axis2_1*16)) + axis3_1)] = B[(((((blockIdx.y*4194304) + cse_var_2) + (axis2_1*16384)) + (axis1_1*16)) + axis3_1)]
              }
            }
          }
        }
      }
    }
    for (ax0: int32, 0, 256) {
      for (ax1: int32, 0, 16384) {
        B.global.shared_1: Buffer(B.global.shared, int8, [4194304], [], scope="shared")[((ax0*16384) + ax1)] = B.global_1[((((floordiv(ax0, 16)*262144) + (floordiv(ax1, 16)*256)) + (floormod(ax0, 16)*16)) + floormod(ax1, 16))]
      }
    }
    attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 256;
    allocate(C.wmma.accumulator: Pointer(wmma.accumulator int32), int32, [4096]), storage_scope = wmma.accumulator;
    allocate(A.global.shared: Pointer(shared int8), int8, [4096]), storage_scope = shared;
    allocate(A.global.shared.wmma.matrix_a: Pointer(wmma.matrix_a int8), int8, [512]), storage_scope = wmma.matrix_a;
    allocate(B.global.shared.wmma.matrix_b: Pointer(wmma.matrix_b int8), int8, [2048]), storage_scope = wmma.matrix_b;
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 64;
    attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;
    attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 2 {
      for (ii.c.outer.init: int32, 0, 2) {
        for (jj.c.outer.init: int32, 0, 8) {
          for (ii.c.inner.init: int32, 0, 16) {
            for (jj.c.inner.init: int32, 0, 16) {
              C.wmma.accumulator_1: Buffer(C.wmma.accumulator, int32, [4096], [], scope="wmma.accumulator")[((((ii.c.outer.init*2048) + (ii.c.inner.init*128)) + (jj.c.outer.init*16)) + jj.c.inner.init)] = 0
            }
          }
        }
      }
      for (rk.outer.outer: int32, 0, 256) {
        for (ax0_1: int32, 0, 64) {
          for (ax1_1: int32, 0, 64) {
            A.global.shared_1: Buffer(A.global.shared, int8, [4096], [], scope="shared")[((ax0_1*64) + ax1_1)] = A.global_1[((((((blockIdx.x*1048576) + (floordiv(ax0_1, 16)*262144)) + (rk.outer.outer*1024)) + (floordiv(ax1_1, 16)*256)) + (floormod(ax0_1, 16)*16)) + floormod(ax1_1, 16))]
          }
        }
        for (rk.outer.inner: int32, 0, 4) {
          for (ax0.outer: int32, 0, 2) {
            for (ax0.inner: int32, 0, 16) {
              for (ax1.inner: int32, 0, 16) {
                A.global.shared.wmma.matrix_a_1: Buffer(A.global.shared.wmma.matrix_a, int8, [512], [], scope="wmma.matrix_a")[(((ax0.outer*256) + (ax0.inner*16)) + ax1.inner)] = A.global.shared_1[(((((threadIdx.y*2048) + (ax0.outer*1024)) + (ax0.inner*64)) + (rk.outer.inner*16)) + ax1.inner)]
              }
            }
          }
          for (ax0.outer_1: int32, 0, 8) {
            for (ax0.inner_1: int32, 0, 16) {
              for (ax1.inner_1: int32, 0, 16) {
                B.global.shared.wmma.matrix_b_1: Buffer(B.global.shared.wmma.matrix_b, int8, [2048], [], scope="wmma.matrix_b")[(((ax0.outer_1*256) + (ax0.inner_1*16)) + ax1.inner_1)] = B.global.shared_1[((((((threadIdx.z*2097152) + (ax0.outer_1*262144)) + (ax0.inner_1*16384)) + (rk.outer.outer*64)) + (rk.outer.inner*16)) + ax1.inner_1)]
              }
            }
          }
          for (ii.c.outer: int32, 0, 2) {
            for (jj.c.outer: int32, 0, 8) {
              for (ii.c.inner: int32, 0, 16) {
                for (jj.c.inner: int32, 0, 16) {
                  for (rk.inner: int32, 0, 16) {
                    let cse_var_3: int32 = ((((ii.c.outer*2048) + (ii.c.inner*128)) + (jj.c.outer*16)) + jj.c.inner)
                    C.wmma.accumulator_1[cse_var_3] = (C.wmma.accumulator_1[cse_var_3] + (cast(int32, A.global.shared.wmma.matrix_a_1[(((ii.c.outer*256) + (ii.c.inner*16)) + rk.inner)])*cast(int32, B.global.shared.wmma.matrix_b_1[(((jj.c.outer*256) + (jj.c.inner*16)) + rk.inner)])))
                  }
                }
              }
            }
          }
        }
      }
      for (axis0.inner: int32, 0, 2) {
        for (axis1.inner: int32, 0, 8) {
          for (axis2_2: int32, 0, 16) {
            for (axis3_2: int32, 0, 16) {
              C.wmma.accumulator.global_1: Buffer(C.wmma.accumulator.global, int32, [268435456], [])[((((((((blockIdx.x*1048576) + (threadIdx.y*524288)) + (axis0.inner*262144)) + (blockIdx.y*4096)) + (threadIdx.z*2048)) + (axis1.inner*256)) + (axis2_2*16)) + axis3_2)] = C.wmma.accumulator_1[((((axis0.inner*2048) + (axis2_2*128)) + (axis1.inner*16)) + axis3_2)]
            }
          }
        }
      }
    }
    for (ii: int32, 0, 16384) {
      for (jj: int32, 0, 16384) {
        C[((ii*16384) + jj)] = C.wmma.accumulator.global_1[((((floordiv(ii, 16)*262144) + (floordiv(jj, 16)*256)) + (floormod(ii, 16)*16)) + floormod(jj, 16))]
      }
    }
  }
}

