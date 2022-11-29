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


def shared_16x16_to_ldmatrix_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


def shared_offset(tx, stride): return stride * (tx % 16) + 8 * (
    tx // 16
)

# convert 4d array to 2d array
def convert4d_to_2d(i, j, k, l):
    return (i * 4 + j, k * 16 + l)


def global_16x32_to_shared_load_16x32(i, j):
    # 0, 0-16 -> 0, 0-16
    # 1, 0-16 -> 1, 0-16
    # 2, 0-16 -> 2, 0-16
    # 3, 0-16 -> 3, 0-16
    """
        re-orgnize the global memory to shared memory access pattern
        key context : 
            j % 16 -> index
            j // 16 
            i % 16 -> index
    """
    thread_id = (i * 32 + j) // 16
    row = thread_id % 16
    col = (j % 16) + (thread_id // 16) * 16
    return row, col


def shared_16x32_to_ldmatrix_32x16_permutation(i, j):
    return (j // 16) * 16 + (i // 8) * 8 + i % 8, j % 16

# for m in range(0, 16):
#     for n in range(0, 16, 1):
#         print(m, n, shared_16x16_to_ldmatrix_32x8_layout(m, n))

def shared_16x16_to_ldmatrix_32x8_permutation(i, j):
    return (j // 8) * 16 + (i // 8) * 8 + i % 8, j % 8


# def shared_16x32_to_ldmatrix_32x16_layout(i, j):
#     # convert (i // 8, j // 16, i % 8, j % 16) to a 2d array
#     return (i * 2 + j // 16, j % 16)


def shared_16x16_to_ldmatrix_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)

for m in range(0, 16):
    for n in range(0, 32, 1):
        print(m * 2 + n // 16, n %
              16, global_16x32_to_shared_load_16x32(m, n))
