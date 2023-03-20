# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(128, 42, 42, 1024), "float16"], W: T.Buffer[(384, 1, 1, 1024), "float16"], Conv: T.Buffer[(225792, 384), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    data_im2col_shared = T.alloc_buffer([14112, 64, 16, 16], dtype="float16", scope="shared")
    data_im2col_shared_warp = T.alloc_buffer([14112, 64, 32, 8], dtype="float16", scope="warp")
    weight_flatten_shared = T.alloc_buffer([24, 64, 16, 16], dtype="float16", scope="shared")
    weight_flatten_shared_warp = T.alloc_buffer([24, 64, 32, 8], dtype="float16", scope="warp")
    Conv_warp = T.alloc_buffer([14112, 24, 32, 8], dtype="float16", scope="warp")
    for x_0_0 in T.thread_binding(882, thread="blockIdx.y"):
        for y_0_0 in T.thread_binding(3, thread="blockIdx.x"):
            for x_0_1 in T.thread_binding(4, thread="threadIdx.y"):
                for y_0_1 in T.thread_binding(2, thread="threadIdx.z"):
                    for x_0_2_init, y_0_2_init, x_1_init, y_1_init in T.grid(4, 4, 16, 16):
                        with T.block("Conv_init"):
                            v_x = T.axis.spatial(225792, x_0_0 * 256 + x_0_1 * 64 + x_0_2_init * 16 + x_1_init)
                            v_y = T.axis.spatial(384, y_0_0 * 128 + y_0_1 * 64 + y_0_2_init * 16 + y_1_init)
                            T.reads()
                            T.writes(Conv_warp[v_x // 16, v_y // 16, v_x % 8 * 4 + v_y % 8 // 2, v_y % 16 // 8 * 4 + v_x % 16 // 8 * 2 + v_y % 2])
                            Conv_warp[v_x // 16, v_y // 16, v_x % 8 * 4 + v_y % 8 // 2, v_y % 16 // 8 * 4 + v_x % 16 // 8 * 2 + v_y % 2] = T.float16(0)
                    for k_0_0 in T.serial(32):
                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(4):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                            with T.block("data_im2col_shared"):
                                                v0 = T.axis.spatial(225792, x_0_0 * 256 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 512 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                v1 = T.axis.spatial(1024, k_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 512 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                T.reads(A[v0 // 1764, v1 // 1024 + v0 % 1764 // 42, v0 % 42, v1 % 1024])
                                                T.writes(data_im2col_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                data_im2col_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = T.if_then_else(0 <= 1 * (v0 % 1764 // 42) + 1 * (v1 // 1024 // 1) and 1 * (v0 % 1764 // 42) + 1 * (v1 // 1024 // 1) < 42 and 0 <= 1 * (v0 % 1764 % 42) + 1 * (v1 // 1024 % 1) and 1 * (v0 % 1764 % 42) + 1 * (v1 // 1024 % 1) < 42, A[v0 // 1764, 1 * (v0 % 1764 // 42) + 1 * (v1 // 1024 // 1) - 0, 1 * (v0 % 1764 % 42) + 1 * (v1 // 1024 % 1) - 0, v1 % 1024], T.float16(0), dtype="float16")
                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(2):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                            with T.block("weight_flatten_shared"):
                                                v0 = T.axis.spatial(384, y_0_0 * 128 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 512 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                v1 = T.axis.spatial(1024, k_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 512 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 512 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                T.reads(W[v0, v1 // 1024, 0, v1 % 1024])
                                                T.writes(weight_flatten_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                weight_flatten_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = W[v0, v1 // 1024 // 1, v1 // 1024 % 1, v1 % 1024]
                        for k_0_1 in T.serial(2):
                            for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(4, 1, 16, 16):
                                with T.block("data_im2col_shared_warp"):
                                    v0 = T.axis.spatial(225792, x_0_0 * 256 + x_0_1 * 64 + ax0_0 * 16 + ax0_1)
                                    v1 = T.axis.spatial(1024, k_0_0 * 32 + k_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                    T.reads(data_im2col_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                    T.writes(data_im2col_shared_warp[v0 // 16, v1 // 16, v0 % 16 * 2 + v1 % 16 // 8, v1 % 8])
                                    data_im2col_shared_warp[v0 // 16, v1 // 16, v0 % 16 * 2 + v1 % 16 // 8, v1 % 8] = data_im2col_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                            for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(4, 1, 16, 16):
                                with T.block("weight_flatten_shared_warp"):
                                    v0 = T.axis.spatial(384, y_0_0 * 128 + y_0_1 * 64 + ax0_0 * 16 + ax0_1)
                                    v1 = T.axis.spatial(1024, k_0_0 * 32 + k_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                    T.reads(weight_flatten_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                    T.writes(weight_flatten_shared_warp[v0 // 16, v1 // 16, v0 % 16 * 2 + v1 % 16 // 8, v1 % 8])
                                    weight_flatten_shared_warp[v0 // 16, v1 // 16, v0 % 16 * 2 + v1 % 16 // 8, v1 % 8] = weight_flatten_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                            for x_0_2, y_0_2, x_1, y_1, k_1 in T.grid(4, 4, 16, 16, 16):
                                with T.block("Conv_update"):
                                    v_x = T.axis.spatial(225792, x_0_0 * 256 + x_0_1 * 64 + x_0_2 * 16 + x_1)
                                    v_y = T.axis.spatial(384, y_0_0 * 128 + y_0_1 * 64 + y_0_2 * 16 + y_1)
                                    v_k = T.axis.reduce(1024, k_0_0 * 32 + k_0_1 * 16 + k_1)
                                    T.reads(Conv_warp[v_x // 16, v_y // 16, v_x % 8 * 4 + v_y % 8 // 2, v_y % 16 // 8 * 4 + v_x % 16 // 8 * 2 + v_y % 2], data_im2col_shared_warp[v_x // 16, v_k // 16, v_x % 16 * 2 + v_k % 16 // 8, v_k % 8], weight_flatten_shared_warp[v_y // 16, v_k // 16, v_y % 16 * 2 + v_k % 16 // 8, v_k % 8])
                                    T.writes(Conv_warp[v_x // 16, v_y // 16, v_x % 8 * 4 + v_y % 8 // 2, v_y % 16 // 8 * 4 + v_x % 16 // 8 * 2 + v_y % 2])
                                    Conv_warp[v_x // 16, v_y // 16, v_x % 8 * 4 + v_y % 8 // 2, v_y % 16 // 8 * 4 + v_x % 16 // 8 * 2 + v_y % 2] = Conv_warp[v_x // 16, v_y // 16, v_x % 8 * 4 + v_y % 8 // 2, v_y % 16 // 8 * 4 + v_x % 16 // 8 * 2 + v_y % 2] + data_im2col_shared_warp[v_x // 16, v_k // 16, v_x % 16 * 2 + v_k % 16 // 8, v_k % 8] * weight_flatten_shared_warp[v_y // 16, v_k // 16, v_y % 16 * 2 + v_k % 16 // 8, v_k % 8]
                    for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(4, 4, 16, 16):
                        with T.block("Conv_warp"):
                            v0 = T.axis.spatial(225792, x_0_0 * 256 + x_0_1 * 64 + ax0_0 * 16 + ax0_1)
                            v1 = T.axis.spatial(384, y_0_0 * 128 + y_0_1 * 64 + ax1_0 * 16 + ax1_1)
                            T.reads(Conv_warp[v0 // 16, v1 // 16, v0 % 8 * 4 + v1 % 8 // 2, v1 % 16 // 8 * 4 + v0 % 16 // 8 * 2 + v1 % 2])
                            T.writes(Conv[v0, v1])
                            Conv[v0, v1] = Conv_warp[v0 // 16, v1 // 16, v0 % 8 * 4 + v1 % 8 // 2, v1 % 16 // 8 * 4 + v0 % 16 // 8 * 2 + v1 % 2]
