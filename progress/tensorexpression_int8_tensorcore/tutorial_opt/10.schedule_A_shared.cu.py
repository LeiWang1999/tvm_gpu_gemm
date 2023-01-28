# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16, 14, 14, 16, 16, 16), "float16"], W: T.Buffer[(3, 3, 16, 32, 16, 16), "float16"], Conv: T.Buffer[(16, 14, 14, 32, 16, 16), "float32"]):
    # function attr dict
    T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
    # var definition
    threadIdx_x = T.env_thread("threadIdx.x")
    threadIdx_z = T.env_thread("threadIdx.z")
    threadIdx_y = T.env_thread("threadIdx.y")
    blockIdx_y = T.env_thread("blockIdx.y")
    blockIdx_z = T.env_thread("blockIdx.z")
    blockIdx_x = T.env_thread("blockIdx.x")
    # buffer definition
    A_1 = T.buffer_decl([12845056], dtype="float16", data=A.data)
    Conv_1 = T.buffer_decl([25690112], dtype="float32", data=Conv.data)
    W_1 = T.buffer_decl([1179648], dtype="float16", data=W.data)
    # body
    W_shared = T.allocate([294912], "float16", "shared")
    W_shared_1 = T.buffer_decl([294912], dtype="float16", data=W_shared, scope="shared")
    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(3, 3, 16, 8, 16, 16):
        cse_var_2: T.int32 = ax3 * 256
        cse_var_1: T.int32 = ax4 * 16
        W_shared_1[ax0 * 98304 + ax1 * 32768 + ax2 * 2048 + cse_var_2 + cse_var_1 + ax5] = W_1[ax0 * 393216 + ax1 * 131072 + ax2 * 8192 + blockIdx_y * 2048 + cse_var_2 + cse_var_1 + ax5]
    T.launch_thread(blockIdx_z, 196)
    Conv_wmma_accumulator = T.allocate([2048], "float32", "wmma.accumulator")
    Conv_wmma_accumulator_1 = T.buffer_decl([2048], dtype="float32", data=Conv_wmma_accumulator, scope="wmma.accumulator")
    Apad_shared = T.allocate([12288], "float16", "shared")
    Apad_shared_1 = T.buffer_decl([12288], dtype="float16", data=Apad_shared, scope="shared")
    Apad_shared_wmma_matrix_a = T.allocate([512], "float16", "wmma.matrix_a")
    Apad_shared_wmma_matrix_a_1 = T.buffer_decl([512], dtype="float16", data=Apad_shared_wmma_matrix_a, scope="wmma.matrix_a")
    W_shared_wmma_matrix_b = T.allocate([1024], "float16", "wmma.matrix_b")
    W_shared_wmma_matrix_b_1 = T.buffer_decl([1024], dtype="float16", data=W_shared_wmma_matrix_b, scope="wmma.matrix_b")
    T.launch_thread(blockIdx_x, 2)
    T.launch_thread(blockIdx_y, 4)
    T.launch_thread(threadIdx_y, 4)
    T.launch_thread(threadIdx_z, 2)
    for n_c_init, o_c_init, nn_c_init, oo_c_init in T.grid(2, 4, 16, 16):
        Conv_wmma_accumulator_1[n_c_init * 1024 + o_c_init * 256 + nn_c_init * 16 + oo_c_init] = T.float32(0)
    for ic_outer, kh in T.grid(8, 3):
        for ax2, ax3, ax4_ax5_fused_outer in T.grid(3, 2, 8):
            cse_var_4: T.int32 = ax3 * 256
            cse_var_3: T.int32 = ax4_ax5_fused_outer * 32
            T.launch_thread(threadIdx_x, 32)
            Apad_shared_1[threadIdx_y * 3072 + threadIdx_z * 1536 + ax2 * 512 + cse_var_4 + cse_var_3 + threadIdx_x] = T.if_then_else(1 <= blockIdx_z // 14 + kh and blockIdx_z // 14 + kh < 15 and 1 <= ax2 + blockIdx_z % 14 and ax2 + blockIdx_z % 14 < 15, A_1[blockIdx_x * 6422528 + threadIdx_y * 1605632 + threadIdx_z * 802816 + kh * 57344 + blockIdx_z * 4096 + ax2 * 4096 + ic_outer * 512 + cse_var_4 + cse_var_3 + threadIdx_x - 61440], T.float16(0), dtype="float16")
        for ic_inner, kw in T.grid(2, 3):
            for ax0, ax4, ax5 in T.grid(2, 16, 16):
                cse_var_5: T.int32 = ax4 * 16
                Apad_shared_wmma_matrix_a_1[ax0 * 256 + cse_var_5 + ax5] = Apad_shared_1[threadIdx_y * 3072 + ax0 * 1536 + kw * 512 + ic_inner * 256 + cse_var_5 + ax5]
            for ax3, ax4, ax5 in T.grid(4, 16, 16):
                cse_var_7: T.int32 = ax3 * 256
                cse_var_6: T.int32 = ax4 * 16
                W_shared_wmma_matrix_b_1[cse_var_7 + cse_var_6 + ax5] = W_shared_1[kh * 98304 + kw * 32768 + ic_outer * 4096 + ic_inner * 2048 + threadIdx_z * 1024 + cse_var_7 + cse_var_6 + ax5]
            for n_c, o_c, nn_c, oo_c, ii in T.grid(2, 4, 16, 16, 16):
                cse_var_10: T.int32 = o_c * 256
                cse_var_9: T.int32 = nn_c * 16
                cse_var_8: T.int32 = n_c * 1024 + cse_var_10 + cse_var_9 + oo_c
                Conv_wmma_accumulator_1[cse_var_8] = Conv_wmma_accumulator_1[cse_var_8] + T.Cast("float32", Apad_shared_wmma_matrix_a_1[n_c * 256 + cse_var_9 + ii]) * T.Cast("float32", W_shared_wmma_matrix_b_1[cse_var_10 + ii * 16 + oo_c])
    for n_inner, o_inner, nn, oo in T.grid(2, 4, 16, 16):
        cse_var_12: T.int32 = o_inner * 256
        cse_var_11: T.int32 = nn * 16
        Conv_1[blockIdx_x * 12845056 + threadIdx_y * 3211264 + n_inner * 1605632 + blockIdx_z * 8192 + blockIdx_y * 2048 + threadIdx_z * 1024 + cse_var_12 + cse_var_11 + oo] = Conv_wmma_accumulator_1[n_inner * 1024 + cse_var_12 + cse_var_11 + oo]
