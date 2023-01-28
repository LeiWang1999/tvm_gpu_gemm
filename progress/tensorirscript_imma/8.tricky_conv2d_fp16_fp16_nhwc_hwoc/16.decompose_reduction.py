# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(16, 14, 14, 16, 16, 16), "float16"], W: T.Buffer[(3, 3, 16, 32, 16, 16), "float16"], Conv: T.Buffer[(16, 14, 14, 32, 16, 16), "float16"]):
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    Apad = T.alloc_buffer([16, 16, 16, 16, 16, 16], dtype="float16")
    Apad_shared = T.alloc_buffer([16, 16, 16, 16, 16, 16], dtype="float16", scope="shared")
    Apad_shared_wmma_matrix_a = T.alloc_buffer([16, 16, 16, 16, 16, 16], dtype="float16", scope="wmma.matrix_a")
    W_shared = T.alloc_buffer([3, 3, 16, 32, 16, 16], dtype="float16", scope="shared")
    W_shared_wmma_matrix_b = T.alloc_buffer([3, 3, 16, 32, 16, 16], dtype="float16", scope="wmma.matrix_b")
    Conv_wmma_accumulator = T.alloc_buffer([16, 14, 14, 32, 16, 16], dtype="float16", scope="wmma.accumulator")
    for n, h, w, i, nn, ii in T.grid(16, 16, 16, 16, 16, 16):
        with T.block("Apad_pad_const"):
            v_n, v_h, v_w, v_i, v_nn, v_ii = T.axis.remap("SSSSSS", [n, h, w, i, nn, ii])
            T.reads()
            T.writes(Apad[v_n, v_h, v_w, v_i, v_nn, v_ii])
            Apad[v_n, v_h, v_w, v_i, v_nn, v_ii] = T.float16(0)
    for n, h, w, i, nn, ii in T.grid(16, 14, 14, 16, 16, 16):
        with T.block("Apad"):
            v_n, v_h, v_w, v_i, v_nn, v_ii = T.axis.remap("SSSSSS", [n, h, w, i, nn, ii])
            T.reads(A[v_n, v_h, v_w, v_i, v_nn, v_ii])
            T.writes(Apad[v_n, v_h + 1, v_w + 1, v_i, v_nn, v_ii])
            Apad[v_n, v_h + 1, v_w + 1, v_i, v_nn, v_ii] = A[v_n, v_h, v_w, v_i, v_nn, v_ii]
    for n_0_0 in T.thread_binding(2, thread="blockIdx.x"):
        for o_0_0 in T.thread_binding(4, thread="blockIdx.y"):
            for n_0_1 in T.thread_binding(4, thread="threadIdx.y"):
                for h, w in T.grid(14, 14):
                    for o_0_1 in T.thread_binding(2, thread="threadIdx.z"):
                        for n_1_init, o_1_init, nn_init, oo_init in T.grid(2, 4, 16, 16):
                            with T.block("Conv_init"):
                                v_n = T.axis.spatial(16, n_0_0 * 8 + n_0_1 * 2 + n_1_init)
                                v_h, v_w = T.axis.remap("SS", [h, w])
                                v_o = T.axis.spatial(32, o_0_0 * 8 + o_0_1 * 4 + o_1_init)
                                v_nn, v_oo = T.axis.remap("SS", [nn_init, oo_init])
                                T.reads()
                                T.writes(Conv_wmma_accumulator[v_n, v_h, v_w, v_o, v_nn, v_oo])
                                Conv_wmma_accumulator[v_n, v_h, v_w, v_o, v_nn, v_oo] = T.float16(0)
                        for ic_0, kh in T.grid(8, 3):
                            for ax0_1_0 in T.thread_binding(2, thread="threadIdx.z"):
                                for ax0_0 in T.thread_binding(4, thread="threadIdx.y"):
                                    for ax3_ax4_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax1, ax2, ax3_ax4_fused_0, ax0_1_1 in T.grid(3, 2, 8, 1):
                                            with T.block("Apad_shared"):
                                                v0 = T.axis.spatial(16, ax0_1_1 + n_0_0 * 8 + ax0_0 * 2 + ax0_1_0)
                                                v1 = T.axis.spatial(16, h + kh)
                                                v2 = T.axis.spatial(16, w + ax1)
                                                v3 = T.axis.spatial(16, ic_0 * 2 + ax2)
                                                v4 = T.axis.spatial(16, (ax3_ax4_fused_0 * 32 + ax3_ax4_fused_1) // 16)
                                                v5 = T.axis.spatial(16, (ax3_ax4_fused_0 * 32 + ax3_ax4_fused_1) % 16)
                                                T.reads(Apad[v0, v1, v2, v3, v4, v5])
                                                T.writes(Apad_shared[v0, v1, v2, v3, v4, v5])
                                                Apad_shared[v0, v1, v2, v3, v4, v5] = Apad[v0, v1, v2, v3, v4, v5]
                            for ax0_ax1_ax2_fused_0 in T.thread_binding(4, thread="threadIdx.y"):
                                for ax0_ax1_ax2_fused_1_0 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_ax1_ax2_fused_1_1, ax3_ax4_fused_0 in T.grid(6, 8):
                                        for ax3_ax4_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                            with T.block("W_shared"):
                                                v0 = T.axis.spatial(3, kh)
                                                v1 = T.axis.spatial(3, (ax0_ax1_ax2_fused_0 * 12 + ax0_ax1_ax2_fused_1_0 * 6 + ax0_ax1_ax2_fused_1_1) // 16)
                                                v2 = T.axis.spatial(16, ic_0 * 2 + (ax0_ax1_ax2_fused_0 * 12 + ax0_ax1_ax2_fused_1_0 * 6 + ax0_ax1_ax2_fused_1_1) % 16 // 8)
                                                v3 = T.axis.spatial(32, o_0_0 * 8 + (ax0_ax1_ax2_fused_0 * 12 + ax0_ax1_ax2_fused_1_0 * 6 + ax0_ax1_ax2_fused_1_1) % 8)
                                                v4 = T.axis.spatial(16, (ax3_ax4_fused_0 * 32 + ax3_ax4_fused_1) // 16)
                                                v5 = T.axis.spatial(16, (ax3_ax4_fused_0 * 32 + ax3_ax4_fused_1) % 16)
                                                T.reads(W[v0, v1, v2, v3, v4, v5])
                                                T.writes(W_shared[v0, v1, v2, v3, v4, v5])
                                                W_shared[v0, v1, v2, v3, v4, v5] = W[v0, v1, v2, v3, v4, v5]
                            for ic_1, kw in T.grid(2, 3):
                                for ax0, ax1, ax2 in T.grid(2, 16, 16):
                                    with T.block("Apad_shared_wmma.matrix_a"):
                                        v0 = T.axis.spatial(16, n_0_0 * 8 + n_0_1 * 2 + ax0)
                                        v1 = T.axis.spatial(16, h + kh)
                                        v2 = T.axis.spatial(16, w + kw)
                                        v3 = T.axis.spatial(16, ic_0 * 2 + ic_1)
                                        v4, v5 = T.axis.remap("SS", [ax1, ax2])
                                        T.reads(Apad_shared[v0, v1, v2, v3, v4, v5])
                                        T.writes(Apad_shared_wmma_matrix_a[v0, v1, v2, v3, v4, v5])
                                        Apad_shared_wmma_matrix_a[v0, v1, v2, v3, v4, v5] = Apad_shared[v0, v1, v2, v3, v4, v5]
                                for ax0, ax1, ax2 in T.grid(4, 16, 16):
                                    with T.block("W_shared_wmma.matrix_b"):
                                        v0, v1 = T.axis.remap("SS", [kh, kw])
                                        v2 = T.axis.spatial(16, ic_0 * 2 + ic_1)
                                        v3 = T.axis.spatial(32, o_0_0 * 8 + o_0_1 * 4 + ax0)
                                        v4, v5 = T.axis.remap("SS", [ax1, ax2])
                                        T.reads(W_shared[v0, v1, v2, v3, v4, v5])
                                        T.writes(W_shared_wmma_matrix_b[v0, v1, v2, v3, v4, v5])
                                        W_shared_wmma_matrix_b[v0, v1, v2, v3, v4, v5] = W_shared[v0, v1, v2, v3, v4, v5]
                                for n_1, o_1, nn, oo, ii in T.grid(2, 4, 16, 16, 16):
                                    with T.block("Conv_update"):
                                        v_n = T.axis.spatial(16, n_0_0 * 8 + n_0_1 * 2 + n_1)
                                        v_h, v_w = T.axis.remap("SS", [h, w])
                                        v_o = T.axis.spatial(32, o_0_0 * 8 + o_0_1 * 4 + o_1)
                                        v_nn, v_oo = T.axis.remap("SS", [nn, oo])
                                        v_ic = T.axis.reduce(16, ic_0 * 2 + ic_1)
                                        v_kh, v_kw, v_ii = T.axis.remap("RRR", [kh, kw, ii])
                                        T.reads(Conv_wmma_accumulator[v_n, v_h, v_w, v_o, v_nn, v_oo], Apad_shared_wmma_matrix_a[v_n, v_h + v_kh, v_w + v_kw, v_ic, v_nn, v_ii], W_shared_wmma_matrix_b[v_kh, v_kw, v_ic, v_o, v_ii, v_oo])
                                        T.writes(Conv_wmma_accumulator[v_n, v_h, v_w, v_o, v_nn, v_oo])
                                        Conv_wmma_accumulator[v_n, v_h, v_w, v_o, v_nn, v_oo] = Conv_wmma_accumulator[v_n, v_h, v_w, v_o, v_nn, v_oo] + Apad_shared_wmma_matrix_a[v_n, v_h + v_kh, v_w + v_kw, v_ic, v_nn, v_ii] * W_shared_wmma_matrix_b[v_kh, v_kw, v_ic, v_o, v_ii, v_oo]
                        for ax0, ax1, ax2, ax3 in T.grid(2, 4, 16, 16):
                            with T.block("Conv_wmma.accumulator"):
                                v0 = T.axis.spatial(16, n_0_0 * 8 + n_0_1 * 2 + ax0)
                                v1, v2 = T.axis.remap("SS", [h, w])
                                v3 = T.axis.spatial(32, o_0_0 * 8 + o_0_1 * 4 + ax1)
                                v4, v5 = T.axis.remap("SS", [ax2, ax3])
                                T.reads(Conv_wmma_accumulator[v0, v1, v2, v3, v4, v5])
                                T.writes(Conv[v0, v1, v2, v3, v4, v5])
                                Conv[v0, v1, v2, v3, v4, v5] = Conv_wmma_accumulator[v0, v1, v2, v3, v4, v5]
