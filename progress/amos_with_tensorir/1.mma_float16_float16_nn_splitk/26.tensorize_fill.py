# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1024, 16384), "float16"], B: T.Buffer[(16384, 1024), "float16"], C: T.Buffer[(1024, 1024), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # var definition
    tx = T.env_thread("threadIdx.x")
    # body
    # with T.block("root")
    TC = T.alloc_buffer([1, 1024, 1024], dtype="float16", scope="shared")
    A_global = T.alloc_buffer([64, 1024, 16, 16], dtype="float16")
    A_global_shared = T.alloc_buffer([64, 1024, 16, 16], dtype="float16", scope="shared")
    A_global_shared_warp = T.alloc_buffer([64, 1024, 32, 8], dtype="float16", scope="warp")
    B_global = T.alloc_buffer([1024, 64, 16, 16], dtype="float16")
    B_global_shared = T.alloc_buffer([1024, 64, 16, 16], dtype="float16", scope="shared")
    B_global_shared_warp = T.alloc_buffer([1024, 64, 32, 8], dtype="float16", scope="warp")
    TC_warp = T.alloc_buffer([1, 64, 64, 32, 8], dtype="float16", scope="warp")
    for ax0, ax1 in T.grid(16384, 1024):
        with T.block("B_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v0, v1])
            T.writes(B_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8])
            B_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8] = B[v0, v1]
    for ax0, ax1 in T.grid(1024, 16384):
        with T.block("A_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8])
            A_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8] = A[v0, v1]
    for sk in T.thread_binding(1, thread="blockIdx.z"):
        for i_0_0 in T.thread_binding(8, thread="blockIdx.y"):
            for j_0_0 in T.thread_binding(4, thread="blockIdx.x"):
                for i_0_1 in T.thread_binding(1, thread="threadIdx.y"):
                    for j_0_1 in T.thread_binding(4, thread="threadIdx.z"):
                        for i_0_2_init, j_0_2_init in T.grid(8, 4):
                            with T.block("B_init_o"):
                                vsk = T.axis.spatial(1, sk)
                                vi_o = T.axis.spatial(64, i_0_0 * 8 + i_0_1 * 8 + i_0_2_init)
                                vj_o = T.axis.spatial(64, j_0_0 * 16 + j_0_1 * 4 + j_0_2_init)
                                T.reads()
                                T.writes(TC_warp[vsk, vi_o, vj_o, 0 : 32, 0 : 8])
                                C_warp = T.match_buffer(TC_warp[vsk, vi_o, vj_o, 0 : 32, 0 : 8], [32, 8], dtype="float16", scope="warp", offset_factor=1)
                                T.launch_thread(tx, 32)
                                T.mma_fill(8, C_warp.data, C_warp.elem_offset, dtype="float16")
                        for k_0_0 in T.serial(512):
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(4):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                with T.block("A_global_shared"):
                                                    v0 = T.axis.spatial(1024, i_0_0 * 128 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 512 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(16384, k_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 512 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 4096 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 1024 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(A_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8])
                                                    T.writes(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = A_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8]
                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_0 in T.thread_binding(1, thread="threadIdx.y"):
                                for ax0_0_ax1_0_ax0_1_ax1_1_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                    for ax0_0_ax1_0_ax0_1_ax1_1_fused_2 in T.serial(8):
                                        for ax0_0_ax1_0_ax0_1_ax1_1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_0_ax1_0_ax0_1_ax1_1_fused_4 in T.vectorized(8):
                                                with T.block("B_global_shared"):
                                                    v0 = T.axis.spatial(16384, k_0_0 * 32 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) // 4096 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 256 // 16)
                                                    v1 = T.axis.spatial(1024, j_0_0 * 256 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 4096 // 256 * 16 + (ax0_0_ax1_0_ax0_1_ax1_1_fused_0 * 8192 + ax0_0_ax1_0_ax0_1_ax1_1_fused_1 * 2048 + ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256 + ax0_0_ax1_0_ax0_1_ax1_1_fused_3 * 8 + ax0_0_ax1_0_ax0_1_ax1_1_fused_4) % 16)
                                                    T.reads(B_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8])
                                                    T.writes(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                                    B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_global[v0 // 16, v1 // 16, v0 % 8 * 2 + v1 % 16 // 8, v0 % 16 // 8 * 8 + v1 % 8]
                            for k_0_1 in T.serial(2):
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(8, 1, 16, 16):
                                    with T.block("A_global_shared_warp"):
                                        v0 = T.axis.spatial(1024, i_0_0 * 128 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(16384, k_0_0 * 32 + k_0_1 * 16 + ax1_0 * 16 + ax1_1)
                                        T.reads(A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        T.writes(A_global_shared_warp[v0 // 16, v1 // 16, v0 % 16 * 2 + v1 % 16 // 8, v1 % 8])
                                        A_global_shared_warp[v0 // 16, v1 // 16, v0 % 16 * 2 + v1 % 16 // 8, v1 % 8] = A_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                                for ax0_0, ax1_0, ax0_1, ax1_1 in T.grid(1, 4, 16, 16):
                                    with T.block("B_global_shared_warp"):
                                        v0 = T.axis.spatial(16384, k_0_0 * 32 + k_0_1 * 16 + ax0_0 * 16 + ax0_1)
                                        v1 = T.axis.spatial(1024, j_0_0 * 256 + j_0_1 * 64 + ax1_0 * 16 + ax1_1)
                                        T.reads(B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
                                        T.writes(B_global_shared_warp[v0 // 16, v1 // 16, v0 % 16 * 2 + v1 % 16 // 8, v1 % 8])
                                        B_global_shared_warp[v0 // 16, v1 // 16, v0 % 16 * 2 + v1 % 16 // 8, v1 % 8] = B_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
                                for i_0_2, j_0_2, i_1, j_1, k_1 in T.grid(8, 4, 16, 16, 16):
                                    with T.block("B_update"):
                                        vsk = T.axis.spatial(1, sk)
                                        vi = T.axis.spatial(1024, i_0_0 * 128 + i_0_1 * 128 + i_0_2 * 16 + i_1)
                                        vj = T.axis.spatial(1024, j_0_0 * 256 + j_0_1 * 64 + j_0_2 * 16 + j_1)
                                        vk = T.axis.reduce(16384, k_0_0 * 32 + k_0_1 * 16 + k_1)
                                        T.reads(TC_warp[vsk, vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2], A_global_shared_warp[vi // 16, vk // 16, vi % 16 * 2 + vk % 16 // 8, vk % 8], B_global_shared_warp[vk // 16, vj // 16, vk % 16 * 2 + vj % 16 // 8, vj % 8])
                                        T.writes(TC_warp[vsk, vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2])
                                        TC_warp[vsk, vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2] = TC_warp[vsk, vi // 16, vj // 16, vi % 8 * 4 + vj % 8 // 2, vj % 16 // 8 * 4 + vi % 16 // 8 * 2 + vj % 2] + A_global_shared_warp[vi // 16, vk // 16, vi % 16 * 2 + vk % 16 // 8, vk % 8] * B_global_shared_warp[vk // 16, vj // 16, vk % 16 * 2 + vj % 16 // 8, vj % 8]
                        for ax0_0 in T.serial(8):
                            for ax1_0, ax0_1, ax1_1 in T.grid(4, 16, 16):
                                with T.block("TC_warp"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(1024, i_0_0 * 128 + ax0_0 * 16 + ax0_1)
                                    v2 = T.axis.spatial(1024, j_0_0 * 256 + j_0_1 * 64 + ax1_0 * 16 + ax1_1)
                                    T.reads(TC_warp[v0, v1 // 16, v2 // 16, v1 % 8 * 4 + v2 % 8 // 2, v2 % 16 // 8 * 4 + v1 % 16 // 8 * 2 + v2 % 2])
                                    T.writes(TC[v0, v1, v2])
                                    TC[v0, v1, v2] = TC_warp[v0, v1 // 16, v2 // 16, v1 % 8 * 4 + v2 % 8 // 2, v2 % 16 // 8 * 4 + v1 % 16 // 8 * 2 + v2 % 2]
                            for ax0, ax1 in T.grid(16, 64):
                                with T.block("C"):
                                    vsk = T.axis.spatial(1, 0)
                                    vi = T.axis.spatial(1024, i_0_0 * 128 + ax0_0 * 16 + ax0)
                                    vj = T.axis.spatial(1024, j_0_0 * 256 + j_0_1 * 64 + ax1)
                                    T.reads(C[vi, vj], TC[vsk, vi, vj])
                                    T.writes()
                                    T.atomic_add(T.address_of(C[vi, vj], dtype="handle"), TC[vsk, vi, vj], dtype="float16")
