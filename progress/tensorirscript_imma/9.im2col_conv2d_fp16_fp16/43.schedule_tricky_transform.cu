#[version = "0.0.5"]
@main = primfn(a: handle, w: handle, conv: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float16), float16, [1, 56, 56, 64], []),
             W: Buffer(W_1: Pointer(global float16), float16, [3, 3, 64, 64], []),
             Conv: Buffer(Conv_1: Pointer(global float16), float16, [1, 3136, 64], [])}
  buffer_map = {a: A, w: W, conv: Conv} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    data_im2colPad_global = alloc_buffer(float16[1, 200, 40, 16, 16])
    data_im2colPad_global_shared = alloc_buffer(float16[1, 200, 40, 16, 16])
    data_im2colPad_global_shared_wmma.matrix_a = alloc_buffer(float16[1, 200, 40, 16, 16])
    weight_flattenPad_global = alloc_buffer(float16[40, 8, 16, 16])
    weight_flattenPad_global_shared = alloc_buffer(float16[40, 8, 16, 16])
    weight_flattenPad_global_shared_wmma.matrix_b = alloc_buffer(float16[40, 8, 16, 16])
    CPad_shared = alloc_buffer(float16[1, 3200, 128])
    CPad_shared_wmma.accumulator = alloc_buffer(float16[1, 200, 8, 16, 16])
     {
      for (ax0: int32, 0, 1) {
        for (ax1_ax2_fused_0: int32, 0, 4) "thread_binding" {
          for (ax1_ax2_fused_1: int32, 0, 2048) "thread_binding" {
            for (ax1_ax2_fused_2: int32, 0, 1) "thread_binding" {
              for (ax1_ax2_fused_3: int32, 0, 128) "thread_binding" {
                for (ax1_ax2_fused_4: int32, 0, 8) "thread_binding" {
                  for (ax1_ax2_fused_5: int32, 0, 1) {
                    for (ax1_ax2_fused_6: int32, 0, 8) "vectorized" {
                      block([1, 3200, 640], "data_im2colPad_global") as [v0, v1, v2] {
                        where((((((((((((ax1_ax2_fused_0*2048) + ax1_ax2_fused_1) + ax1_ax2_fused_2)*128) + ax1_ax2_fused_3)*8) + ax1_ax2_fused_4) + ax1_ax2_fused_5)*8) + ax1_ax2_fused_6) < 2048000))
                        bind(v0, ax0)
                        bind(v1, floordiv((((((((ax1_ax2_fused_0*16777216) + (ax1_ax2_fused_1*8192)) + (ax1_ax2_fused_2*8192)) + (ax1_ax2_fused_3*64)) + (ax1_ax2_fused_4*8)) + (ax1_ax2_fused_5*8)) + ax1_ax2_fused_6), 640))
                        bind(v2, floormod((((((((ax1_ax2_fused_0*16777216) + (ax1_ax2_fused_1*8192)) + (ax1_ax2_fused_2*8192)) + (ax1_ax2_fused_3*64)) + (ax1_ax2_fused_4*8)) + (ax1_ax2_fused_5*8)) + ax1_ax2_fused_6), 640))
                        tir.reads([A[v0, ((floordiv(v2, 192) + floordiv(v1, 56)) - 1), ((floordiv(floormod(v2, 192), 64) + floormod(v1, 56)) - 1), floormod(v2, 64)]])
                        tir.writes([data_im2colPad_global[v0, floordiv(v1, 16), floordiv(v2, 16), floormod(v1, 16), floormod(v2, 16)]])
                        data_im2colPad_global[v0, floordiv(v1, 16), floordiv(v2, 16), floormod(v1, 16), floormod(v2, 16)] = @tir.if_then_else(((v1 < 3136) && (v2 < 576)), @tir.if_then_else(((((1 <= ((1*floordiv(v1, 56)) + (1*floordiv(floordiv(v2, 64), 3)))) && (((1*floordiv(v1, 56)) + (1*floordiv(floordiv(v2, 64), 3))) < 57)) && (1 <= ((1*floormod(v1, 56)) + (1*floormod(floordiv(v2, 64), 3))))) && (((1*floormod(v1, 56)) + (1*floormod(floordiv(v2, 64), 3))) < 57)), A[v0, (((1*floordiv(v1, 56)) + (1*floordiv(floordiv(v2, 64), 3))) - 1), (((1*floormod(v1, 56)) + (1*floormod(floordiv(v2, 64), 3))) - 1), floormod(v2, 64)], 0f16, dtype=float16), 0f16, dtype=float16)
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (ax0_ax1_fused_0: int32, 0, 4) "thread_binding" {
        for (ax0_ax1_fused_1: int32, 0, 2048) "thread_binding" {
          for (ax0_ax1_fused_2: int32, 0, 1) "thread_binding" {
            for (ax0_ax1_fused_3: int32, 0, 128) "thread_binding" {
              for (ax0_ax1_fused_4: int32, 0, 8) "thread_binding" {
                for (ax0_ax1_fused_5: int32, 0, 1) {
                  for (ax0_ax1_fused_6: int32, 0, 8) "vectorized" {
                    block([640, 128], "weight_flattenPad_global") as [v0_1, v1_1] {
                      where((((((((((((ax0_ax1_fused_0*2048) + ax0_ax1_fused_1) + ax0_ax1_fused_2)*128) + ax0_ax1_fused_3)*8) + ax0_ax1_fused_4) + ax0_ax1_fused_5)*8) + ax0_ax1_fused_6) < 81920))
                      bind(v0_1, floordiv((((((((ax0_ax1_fused_0*16777216) + (ax0_ax1_fused_1*8192)) + (ax0_ax1_fused_2*8192)) + (ax0_ax1_fused_3*64)) + (ax0_ax1_fused_4*8)) + (ax0_ax1_fused_5*8)) + ax0_ax1_fused_6), 128))
                      bind(v1_1, floormod((((((((ax0_ax1_fused_0*16777216) + (ax0_ax1_fused_1*8192)) + (ax0_ax1_fused_2*8192)) + (ax0_ax1_fused_3*64)) + (ax0_ax1_fused_4*8)) + (ax0_ax1_fused_5*8)) + ax0_ax1_fused_6), 128))
                      tir.reads([W[floordiv(v0_1, 192), floordiv(floormod(v0_1, 192), 64), floormod(v0_1, 64), v1_1]])
                      tir.writes([weight_flattenPad_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)]])
                      weight_flattenPad_global[floordiv(v0_1, 16), floordiv(v1_1, 16), floormod(v0_1, 16), floormod(v1_1, 16)] = @tir.if_then_else(((v0_1 < 576) && (v1_1 < 64)), W[floordiv(floordiv(v0_1, 64), 3), floormod(floordiv(v0_1, 64), 3), floormod(v0_1, 64), v1_1], 0f16, dtype=float16)
                  }
                }
              }
            }
          }
        }
      }
      for (n: int32, 0, 1) {
        for (x_0_0: int32, 0, 25) "thread_binding" {
          for (y_0_0: int32, 0, 1) "thread_binding" {
            for (x_0_1: int32, 0, 1) "thread_binding" {
              for (y_0_1: int32, 0, 4) "thread_binding" {
                for (x_0_2_init: int32, 0, 8) {
                  for (y_0_2_init: int32, 0, 2) {
                    block([1, 200, 8], "Conv_init_o") as [v_n, v_x_o, v_y_o] {
                      bind(v_n, n)
                      bind(v_x_o, (((x_0_0*8) + (x_0_1*8)) + x_0_2_init))
                      bind(v_y_o, (((y_0_0*8) + (y_0_1*2)) + y_0_2_init))
                      tir.reads([])
                      tir.writes([CPad_shared_wmma.accumulator[v_n, v_x_o, v_y_o, 0:16, 0:16]])
                      C = match_buffer(CPad_shared_wmma.accumulator[v_n, v_x_o, v_y_o, 0:16, 0:16])
                      @tir.tvm_fill_fragment(C_1: Pointer(wmma.accumulator float16), 16, 16, 16, ((floordiv(floordiv(elem_offset: int32, C_s0: int32), 16)*floordiv(C_s0, 16)) + floordiv(floormod(elem_offset, C_s0), 16)), 0f32, dtype=handle)
                  }
                }
                for (k_0_0: int32, 0, 20) {
                  for (ax0_0_ax1_0_ax0_1_ax1_1_fused_0: int32, 0, 1) "thread_binding" {
                    for (ax0_0_ax1_0_ax0_1_ax1_1_fused_1: int32, 0, 4) "thread_binding" {
                      for (ax0_0_ax1_0_ax0_1_ax1_1_fused_2: int32, 0, 4) {
                        for (ax0_0_ax1_0_ax0_1_ax1_1_fused_3: int32, 0, 32) "thread_binding" {
                          for (ax0_0_ax1_0_ax0_1_ax1_1_fused_4: int32, 0, 8) "vectorized" {
                            block([1, 3200, 640], "data_im2colPad_global_shared") as [v0_2, v1_2, v2_1] {
                              bind(v0_2, 0)
                              bind(v1_2, (((x_0_0*128) + (floordiv((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 512)*16)) + floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 256), 16)))
                              bind(v2_1, (((k_0_0*32) + (floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 512), 256)*16)) + floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4), 16)))
                              tir.reads([data_im2colPad_global[v0_2, floordiv(v1_2, 16), floordiv(v2_1, 16), floormod(v1_2, 16), floormod(v2_1, 16)]])
                              tir.writes([data_im2colPad_global_shared[v0_2, floordiv(v1_2, 16), floordiv(v2_1, 16), floormod(v1_2, 16), floormod(v2_1, 16)]])
                              data_im2colPad_global_shared[v0_2, floordiv(v1_2, 16), floordiv(v2_1, 16), floormod(v1_2, 16), floormod(v2_1, 16)] = data_im2colPad_global[v0_2, floordiv(v1_2, 16), floordiv(v2_1, 16), floormod(v1_2, 16), floormod(v2_1, 16)]
                          }
                        }
                      }
                    }
                  }
                  for (ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1: int32, 0, 1) "thread_binding" {
                    for (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1: int32, 0, 4) "thread_binding" {
                      for (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1: int32, 0, 4) {
                        for (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1: int32, 0, 32) "thread_binding" {
                          for (ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1: int32, 0, 8) "vectorized" {
                            block([640, 128], "weight_flattenPad_global_shared") as [v0_3, v1_3] {
                              bind(v0_3, (((k_0_0*32) + (floordiv((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 2048)*16)) + floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 256), 16)))
                              bind(v1_3, ((floordiv(floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 2048), 256)*16) + floormod((((((ax0_0_ax1_0_ax0_1_ax1_1_fused_0_1*4096) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_1_1*1024)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1*256)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_3_1*8)) + ax0_0_ax1_0_ax0_1_ax1_1_fused_4_1), 16)))
                              tir.reads([weight_flattenPad_global[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
                              tir.writes([weight_flattenPad_global_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]])
                              weight_flattenPad_global_shared[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)] = weight_flattenPad_global[floordiv(v0_3, 16), floordiv(v1_3, 16), floormod(v0_3, 16), floormod(v1_3, 16)]
                          }
                        }
                      }
                    }
                  }
                  for (k_0_1: int32, 0, 2) {
                    for (ax0_0: int32, 0, 8) {
                      for (ax1_0: int32, 0, 1) {
                        block([1, 200, 40], "data_im2colPad_global_shared_wmma.matrix_a_o") as [v0_4, v1_o, v2_o] {
                          bind(v0_4, 0)
                          bind(v1_o, ((x_0_0*8) + ax0_0))
                          bind(v2_o, (((k_0_0*2) + k_0_1) + ax1_0))
                          tir.reads([data_im2colPad_global_shared[v0_4, v1_o, v2_o, 0:16, 0:16]])
                          tir.writes([data_im2colPad_global_shared_wmma.matrix_a[v0_4, v1_o, v2_o, 0:16, 0:16]])
                          A_2 = match_buffer(data_im2colPad_global_shared[v0_4, v1_o, v2_o, 0:16, 0:16])
                          C_2 = match_buffer(data_im2colPad_global_shared_wmma.matrix_a[v0_4, v1_o, v2_o, 0:16, 0:16])
                          @tir.tvm_load_matrix_sync(C_3: Pointer(wmma.matrix_a float16), 16, 16, 16, ((floordiv(floordiv(elem_offset_1: int32, C_s0_1: int32), 16)*floordiv(C_s0_1, 16)) + floordiv(floormod(elem_offset_1, C_s0_1), 16)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), A_3: Pointer(shared float16), elem_offset_2: int32, (A_s0: int32*16), 1, dtype=handle), A_s0, "row_major", dtype=handle)
                      }
                    }
                    for (ax0_0_1: int32, 0, 1) {
                      for (ax1_0_1: int32, 0, 2) {
                        block([40, 8], "weight_flattenPad_global_shared_wmma.matrix_b_o") as [v0_o, v1_o_1] {
                          bind(v0_o, (((k_0_0*2) + k_0_1) + ax0_0_1))
                          bind(v1_o_1, ((y_0_1*2) + ax1_0_1))
                          tir.reads([weight_flattenPad_global_shared[v0_o, v1_o_1, 0:16, 0:16]])
                          tir.writes([weight_flattenPad_global_shared_wmma.matrix_b[v0_o, v1_o_1, 0:16, 0:16]])
                          A_4 = match_buffer(weight_flattenPad_global_shared[v0_o, v1_o_1, 0:16, 0:16])
                          C_4 = match_buffer(weight_flattenPad_global_shared_wmma.matrix_b[v0_o, v1_o_1, 0:16, 0:16])
                          @tir.tvm_load_matrix_sync(C_5: Pointer(wmma.matrix_b float16), 16, 16, 16, ((floordiv(floordiv(elem_offset_3: int32, C_s0_2: int32), 16)*floordiv(C_s0_2, 16)) + floordiv(floormod(elem_offset_3, C_s0_2), 16)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), A_5: Pointer(shared float16), elem_offset_4: int32, (A_s0_1: int32*16), 1, dtype=handle), A_s0_1, "row_major", dtype=handle)
                      }
                    }
                    for (x_0_2: int32, 0, 8) {
                      for (y_0_2: int32, 0, 2) {
                        block([1, 200, 8, tir.reduce_axis(0, 40)], "Conv_update_o") as [v_n_1, v_x_o_1, v_y_o_1, v_k_o] {
                          bind(v_n_1, n)
                          bind(v_x_o_1, (((x_0_0*8) + (x_0_1*8)) + x_0_2))
                          bind(v_y_o_1, (((y_0_0*8) + (y_0_1*2)) + y_0_2))
                          bind(v_k_o, ((k_0_0*2) + k_0_1))
                          tir.reads([CPad_shared_wmma.accumulator[v_n_1, v_x_o_1, v_y_o_1, 0:16, 0:16], data_im2colPad_global_shared_wmma.matrix_a[v_n_1, v_x_o_1, v_k_o, 0:16, 0:16], weight_flattenPad_global_shared_wmma.matrix_b[v_k_o, v_y_o_1, 0:16, 0:16]])
                          tir.writes([CPad_shared_wmma.accumulator[v_n_1, v_x_o_1, v_y_o_1, 0:16, 0:16]])
                          A_6 = match_buffer(data_im2colPad_global_shared_wmma.matrix_a[v_n_1, v_x_o_1, v_k_o, 0:16, 0:16])
                          B = match_buffer(weight_flattenPad_global_shared_wmma.matrix_b[v_k_o, v_y_o_1, 0:16, 0:16])
                          C_6 = match_buffer(CPad_shared_wmma.accumulator[v_n_1, v_x_o_1, v_y_o_1, 0:16, 0:16])
                          @tir.tvm_mma_sync(C_7: Pointer(wmma.accumulator float16), ((floordiv(floordiv(elem_offset_5: int32, C_s0_3: int32), 16)*floordiv(C_s0_3, 16)) + floordiv(floormod(elem_offset_5, C_s0_3), 16)), A_7: Pointer(wmma.matrix_a float16), ((floordiv(floordiv(elem_offset_6: int32, A_s0_2: int32), 16)*floordiv(A_s0_2, 16)) + floordiv(floormod(elem_offset_6, A_s0_2), 16)), B_1: Pointer(wmma.matrix_b float16), ((floordiv(floordiv(elem_offset_7: int32, B_s0: int32), 16)*floordiv(B_s0, 16)) + floordiv(floormod(elem_offset_7, B_s0), 16)), C_7, ((floordiv(floordiv(elem_offset_5, C_s0_3), 16)*floordiv(C_s0_3, 16)) + floordiv(floormod(elem_offset_5, C_s0_3), 16)), dtype=handle)
                      }
                    }
                  }
                }
                for (ax0_0_2: int32, 0, 8) {
                  for (ax1_0_2: int32, 0, 2) {
                    block([1, 200, 8], "CPad_shared_wmma.accumulator_o") as [v0_5, v1_o_2, v2_o_1] {
                      bind(v0_5, 0)
                      bind(v1_o_2, ((x_0_0*8) + ax0_0_2))
                      bind(v2_o_1, ((y_0_1*2) + ax1_0_2))
                      tir.reads([CPad_shared_wmma.accumulator[v0_5, v1_o_2, v2_o_1, 0:16, 0:16]])
                      tir.writes([CPad_shared[v0_5, (v1_o_2*16):((v1_o_2*16) + 16), (v2_o_1*16):((v2_o_1*16) + 16)]])
                      A_8 = match_buffer(CPad_shared_wmma.accumulator[v0_5, v1_o_2, v2_o_1, 0:16, 0:16])
                      C_8 = match_buffer(CPad_shared[v0_5, (v1_o_2*16):((v1_o_2*16) + 16), (v2_o_1*16):((v2_o_1*16) + 16)])
                      @tir.tvm_store_matrix_sync(A_9: Pointer(wmma.accumulator float16), 16, 16, 16, ((floordiv(floordiv(elem_offset_8: int32, A_s0_3: int32), 16)*floordiv(A_s0_3, 16)) + floordiv(floormod(elem_offset_8, A_s0_3), 16)), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=float16), C_9: Pointer(shared float16), elem_offset_9: int32, (C_s0_4: int32*16), 2, dtype=handle), C_s0_4, "row_major", dtype=handle)
                  }
                  for (ax0_ax1_fused_0_1: int32, 0, 4) {
                    for (ax0_ax1_fused_1_1: int32, 0, 4) "thread_binding" {
                      for (ax0_ax1_fused_2_1: int32, 0, 1) "thread_binding" {
                        for (ax0_ax1_fused_3_1: int32, 0, 32) "thread_binding" {
                          block([1, 3200, 128], "CPad_shared") as [v0_6, v1_4, v2_2] {
                            bind(v0_6, 0)
                            bind(v1_4, (((x_0_0*128) + (ax0_0_2*16)) + floordiv(((((ax0_ax1_fused_0_1*128) + (ax0_ax1_fused_1_1*32)) + (ax0_ax1_fused_2_1*32)) + ax0_ax1_fused_3_1), 32)))
                            bind(v2_2, ((y_0_1*32) + floormod(((((ax0_ax1_fused_0_1*128) + (ax0_ax1_fused_1_1*32)) + (ax0_ax1_fused_2_1*32)) + ax0_ax1_fused_3_1), 32)))
                            tir.reads([CPad_shared[v0_6, v1_4, v2_2]])
                            tir.writes([Conv[v0_6, v1_4, v2_2]])
                            Conv[v0_6, v1_4, v2_2] = CPad_shared[v0_6, v1_4, v2_2]
                        }
                      }
                    }
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