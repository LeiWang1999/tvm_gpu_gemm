from tvm.tir.tensor_intrin.cuda import shared_32x16_to_ldmatrix_32x16_layout

# do permutation for a matrix, and each sub-matrix from 8x64 to 4x128
# each threads writes 4 elements to shared memory, in this permutaion, we have 32 threads in a single warp


def cuda_golden_permutation(i, j):
    def shared_8x64_to_free_4x128_layout(i, j):
        warp_size = 32
        elements_per_thread = 16
        element_id = (i << 6) + j
        thread_id = element_id // elements_per_thread
        element_id_in_thread = element_id % elements_per_thread
        lane_id = thread_id % warp_size
        shared_c = lane_id % 8
        shared_s = lane_id // 8
        shared_row = (shared_c & 1) | ((shared_c >> 1) & 2)
        shared_col = ((shared_c << 1) & 4) | shared_s ^ shared_row
        return (shared_row, shared_col* elements_per_thread + element_id_in_thread)
    return (i // 8, j // 64, *shared_8x64_to_free_4x128_layout(i % 8, j % 64))


def cuda_golden_permutation_v2(i, j):
    def shared_8x64_to_free_4x128_layout(i, j):
        warp_size = 32
        elements_per_thread = 16
        element_id = i * 64 + j
        # 64 / 16 = 4 => (i // 2) + (i % 2) * 4 + (j // 16) * 8
        thread_id = element_id // elements_per_thread
        element_id_in_thread = element_id % elements_per_thread
        lane_id = thread_id % warp_size
        shared_c = lane_id % 8
        shared_s = lane_id // 8
        shared_row = (shared_c & 1) | ((shared_c >> 1) & 2)
        shared_col = ((shared_c << 1) & 4) | shared_s ^ shared_row
        # print(lane_id, shared_c, shared_s, shared_row, shared_col, element_id_in_thread)
        return (shared_row, shared_col * elements_per_thread + element_id_in_thread)
    return (i // 8, j // 64, *shared_8x64_to_free_4x128_layout(i % 8, j % 64))


for m in range(120, 128):
    for n in range(0, 64, 16):
        print(m, n, cuda_golden_permutation(m, n))
