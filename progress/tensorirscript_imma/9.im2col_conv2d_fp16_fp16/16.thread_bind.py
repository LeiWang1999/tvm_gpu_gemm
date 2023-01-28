# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1, 224, 224, 256), "float16"], W: T.Buffer[(7, 7, 256, 512), "float16"], Conv: T.Buffer[(1, 48400, 512), "float16"]):
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    data_im2col_global = T.alloc_buffer([1, 3025, 784, 16, 16], dtype="float16")
    data_im2col_global_shared = T.alloc_buffer([1, 3025, 784, 16, 16], dtype="float16", scope="shared")
    data_im2col_global_shared_wmma_matrix_a = T.alloc_buffer([1, 3025, 784, 16, 16], dtype="float16", scope="wmma.matrix_a")
    weight_flatten_global = T.alloc_buffer([784, 32, 16, 16], dtype="float16")
    weight_flatten_global_shared = T.alloc_buffer([784, 32, 16, 16], dtype="float16", scope="shared")
    weight_flatten_global_shared_wmma_matrix_b = T.alloc_buffer([784, 32, 16, 16], dtype="float16", scope="wmma.matrix_b")
    Conv_wmma_accumulator = T.alloc_buffer([1, 3025, 32, 16, 16], dtype="float16", scope="wmma.accumulator")
    for ax0, ax1, ax2 in T.grid(1, 48400, 12544):
        with T.block("data_im2col_global"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v0, v2 // 1792 + v1 // 220 - 1, v2 % 1792 // 256 + v1 % 220 - 1, v2 % 256])
            T.writes(data_im2col_global[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            data_im2col_global[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16] = T.if_then_else(1 <= v2 // 1792 + v1 // 220 and v2 // 1792 + v1 // 220 < 225 and 1 <= v2 % 1792 // 256 + v1 % 220 and v2 % 1792 // 256 + v1 % 220 < 225, A[v0, v2 // 1792 + v1 // 220 - 1, v2 % 1792 // 256 + v1 % 220 - 1, v2 % 256], T.float16(0), dtype="float16")
    for ax0, ax1, ax2 in T.grid(1, 48400, 12544):
        with T.block("data_im2col_global_shared"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(data_im2col_global[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            T.writes(data_im2col_global_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            data_im2col_global_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16] = data_im2col_global[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16]
    for ax0, ax1, ax2 in T.grid(1, 48400, 12544):
        with T.block("data_im2col_global_shared_wmma.matrix_a"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(data_im2col_global_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            T.writes(data_im2col_global_shared_wmma_matrix_a[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            data_im2col_global_shared_wmma_matrix_a[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16] = data_im2col_global_shared[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16]
    for ax0, ax1 in T.grid(12544, 512):
        with T.block("weight_flatten_global"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(W[v0 // 1792, v0 % 1792 // 256, v0 % 256, v1])
            T.writes(weight_flatten_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            weight_flatten_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = W[v0 // 1792, v0 % 1792 // 256, v0 % 256, v1]
    for ax0, ax1 in T.grid(12544, 512):
        with T.block("weight_flatten_global_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(weight_flatten_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(weight_flatten_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            weight_flatten_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = weight_flatten_global[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for ax0, ax1 in T.grid(12544, 512):
        with T.block("weight_flatten_global_shared_wmma.matrix_b"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(weight_flatten_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            T.writes(weight_flatten_global_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            weight_flatten_global_shared_wmma_matrix_b[v0 // 16, v1 // 16, v0 % 16, v1 % 16] = weight_flatten_global_shared[v0 // 16, v1 // 16, v0 % 16, v1 % 16]
    for n in T.serial(1):
        for x_0_0 in T.thread_binding(3025, thread="blockIdx.y"):
            for y_0_0_0 in T.thread_binding(2, thread="blockIdx.z"):
                for y_0_0_1 in T.thread_binding(16, thread="blockIdx.x"):
                    for x_0_1 in T.thread_binding(1, thread="threadIdx.y"):
                        for y_0_1 in T.thread_binding(1, thread="threadIdx.z"):
                            for k_0_0, k_0_1, x_0_2, y_0_2, x_1, y_1, k_1 in T.grid(196, 4, 1, 1, 16, 16, 16):
                                with T.block("Conv"):
                                    v_n = T.axis.spatial(1, n)
                                    v_x = T.axis.spatial(48400, x_0_0 * 16 + x_0_1 * 16 + x_0_2 * 16 + x_1)
                                    v_y = T.axis.spatial(512, y_0_0_0 * 256 + y_0_0_1 * 16 + y_0_1 * 16 + y_0_2 * 16 + y_1)
                                    v_k = T.axis.reduce(12544, k_0_0 * 64 + k_0_1 * 16 + k_1)
                                    T.reads(data_im2col_global_shared_wmma_matrix_a[v_n, v_x // 16, v_k // 16, v_x % 16, v_k % 16], weight_flatten_global_shared_wmma_matrix_b[v_k // 16, v_y // 16, v_k % 16, v_y % 16])
                                    T.writes(Conv_wmma_accumulator[v_n, v_x // 16, v_y // 16, v_x % 16, v_y % 16])
                                    with T.init():
                                        Conv_wmma_accumulator[v_n, v_x // 16, v_y // 16, v_x % 16, v_y % 16] = T.float16(0)
                                    Conv_wmma_accumulator[v_n, v_x // 16, v_y // 16, v_x % 16, v_y % 16] = Conv_wmma_accumulator[v_n, v_x // 16, v_y // 16, v_x % 16, v_y % 16] + data_im2col_global_shared_wmma_matrix_a[v_n, v_x // 16, v_k // 16, v_x % 16, v_k % 16] * weight_flatten_global_shared_wmma_matrix_b[v_k // 16, v_y // 16, v_k % 16, v_y % 16]
    for ax0, ax1, ax2 in T.grid(1, 48400, 512):
        with T.block("Conv_wmma.accumulator"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(Conv_wmma_accumulator[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16])
            T.writes(Conv[v0, v1, v2])
            Conv[v0, v1, v2] = Conv_wmma_accumulator[v0, v1 // 16, v2 // 16, v1 % 16, v2 % 16]
