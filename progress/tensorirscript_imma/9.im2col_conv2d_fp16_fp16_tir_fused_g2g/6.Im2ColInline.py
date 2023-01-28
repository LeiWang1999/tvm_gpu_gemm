# from tvm.script import tir as T
@T.prim_func
def func(A: T.Buffer[(1, 56, 56, 64), "float16"], W: T.Buffer[(3, 3, 64, 64), "float16"], Conv: T.Buffer[(1, 3136, 64), "float16"]):
    # function attr dict
    T.func_attr({"tir.noalias": True, "global_symbol": "main"})
    # body
    # with T.block("root")
    weight_flatten = T.alloc_buffer([576, 64], dtype="float16")
    data_im2colPad = T.alloc_buffer([1, 3136, 576], dtype="float16")
    weight_flattenPad = T.alloc_buffer([576, 64], dtype="float16")
    CPad = T.alloc_buffer([1, 3136, 64], dtype="float16")
    data_im2colPad_shared = T.alloc_buffer([1, 3136, 576], dtype="float16", scope="shared")
    data_im2colPad_shared_wmma_matrix_a = T.alloc_buffer([1, 3136, 576], dtype="float16", scope="wmma.matrix_a")
    weight_flattenPad_shared = T.alloc_buffer([576, 64], dtype="float16", scope="shared")
    weight_flattenPad_shared_wmma_matrix_b = T.alloc_buffer([576, 64], dtype="float16", scope="wmma.matrix_b")
    CPad_shared = T.alloc_buffer([1, 3136, 64], dtype="float16", scope="shared")
    CPad_shared_wmma_accumulator = T.alloc_buffer([1, 3136, 64], dtype="float16", scope="wmma.accumulator")
    for x, y in T.grid(576, 64):
        with T.block("weight_flatten"):
            v_x, v_y = T.axis.remap("SS", [x, y])
            T.reads(W[v_x // 192, v_x % 192 // 64, v_x % 64, v_y])
            T.writes(weight_flatten[v_x, v_y])
            weight_flatten[v_x, v_y] = W[v_x // 64 // 3, v_x // 64 % 3, v_x % 64, v_y]
    for n, i, k in T.grid(1, 3136, 576):
        with T.block("data_im2colPad"):
            vn, vi, vk = T.axis.remap("SSS", [n, i, k])
            T.reads(A[vn, vk // 192 + vi // 56 - 1, vk % 192 // 64 + vi % 56 - 1, vk % 64])
            T.writes(data_im2colPad[vn, vi, vk])
            data_im2colPad[vn, vi, vk] = T.if_then_else(vi < 3136 and vk < 576, T.if_then_else(1 <= 1 * (vi // 56) + 1 * (vk // 64 // 3) and 1 * (vi // 56) + 1 * (vk // 64 // 3) < 57 and 1 <= 1 * (vi % 56) + 1 * (vk // 64 % 3) and 1 * (vi % 56) + 1 * (vk // 64 % 3) < 57, A[vn, 1 * (vi // 56) + 1 * (vk // 64 // 3) - 1, 1 * (vi % 56) + 1 * (vk // 64 % 3) - 1, vk % 64], T.float16(0), dtype="float16"), T.float16(0), dtype="float16")
    for k, j in T.grid(576, 64):
        with T.block("weight_flattenPad"):
            vk, vj = T.axis.remap("SS", [k, j])
            T.reads(weight_flatten[vk, vj])
            T.writes(weight_flattenPad[vk, vj])
            weight_flattenPad[vk, vj] = T.if_then_else(vk < 576 and vj < 64, weight_flatten[vk, vj], T.float16(0), dtype="float16")
    for ax0, ax1, ax2 in T.grid(1, 3136, 576):
        with T.block("data_im2colPad_shared"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(data_im2colPad[v0, v1, v2])
            T.writes(data_im2colPad_shared[v0, v1, v2])
            data_im2colPad_shared[v0, v1, v2] = data_im2colPad[v0, v1, v2]
    for ax0, ax1, ax2 in T.grid(1, 3136, 576):
        with T.block("data_im2colPad_shared_wmma.matrix_a"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(data_im2colPad_shared[v0, v1, v2])
            T.writes(data_im2colPad_shared_wmma_matrix_a[v0, v1, v2])
            data_im2colPad_shared_wmma_matrix_a[v0, v1, v2] = data_im2colPad_shared[v0, v1, v2]
    for ax0, ax1 in T.grid(576, 64):
        with T.block("weight_flattenPad_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(weight_flattenPad[v0, v1])
            T.writes(weight_flattenPad_shared[v0, v1])
            weight_flattenPad_shared[v0, v1] = weight_flattenPad[v0, v1]
    for ax0, ax1 in T.grid(576, 64):
        with T.block("weight_flattenPad_shared_wmma.matrix_b"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(weight_flattenPad_shared[v0, v1])
            T.writes(weight_flattenPad_shared_wmma_matrix_b[v0, v1])
            weight_flattenPad_shared_wmma_matrix_b[v0, v1] = weight_flattenPad_shared[v0, v1]
    for n, x, y, k in T.grid(1, 3136, 64, 576):
        with T.block("Conv"):
            v_n, v_x, v_y, v_k = T.axis.remap("SSSR", [n, x, y, k])
            T.reads(data_im2colPad_shared_wmma_matrix_a[v_n, v_x, v_k], weight_flattenPad_shared_wmma_matrix_b[v_k, v_y])
            T.writes(CPad_shared_wmma_accumulator[v_n, v_x, v_y])
            with T.init():
                CPad_shared_wmma_accumulator[v_n, v_x, v_y] = T.float16(0)
            CPad_shared_wmma_accumulator[v_n, v_x, v_y] = CPad_shared_wmma_accumulator[v_n, v_x, v_y] + data_im2colPad_shared_wmma_matrix_a[v_n, v_x, v_k] * weight_flattenPad_shared_wmma_matrix_b[v_k, v_y]
    for ax0, ax1, ax2 in T.grid(1, 3136, 64):
        with T.block("CPad_shared_wmma.accumulator"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(CPad_shared_wmma_accumulator[v0, v1, v2])
            T.writes(CPad_shared[v0, v1, v2])
            CPad_shared[v0, v1, v2] = CPad_shared_wmma_accumulator[v0, v1, v2]
    for ax0, ax1, ax2 in T.grid(1, 3136, 64):
        with T.block("CPad_shared"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(CPad_shared[v0, v1, v2])
            T.writes(CPad[v0, v1, v2])
            CPad[v0, v1, v2] = CPad_shared[v0, v1, v2]
    for n, i, j in T.grid(1, 3136, 64):
        with T.block("CPad"):
            vn, vi, vj = T.axis.remap("SSS", [n, i, j])
            T.reads(CPad[vn, vi, vj])
            T.writes(Conv[vn, vi, vj])
            Conv[vn, vi, vj] = CPad[vn, vi, vj]
