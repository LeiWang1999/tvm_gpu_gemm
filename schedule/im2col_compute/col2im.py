# The sizes of inputs and filters
batch_size = 128
height = 42
width = 42
in_channels = 1024
out_channels = 384
kernel_h = 1
kernel_w = 1
pad_h = 0
pad_w = 0
stride_h = 1
stride_w = 1
dilation_h = 1
dilation_w = 1
output_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
output_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
print("output_height: ", output_height)
print("output_width: ", output_width)
wmma_m = 16
wmma_n = 16
wmma_k = 32

def A_global_16x32_to_shared_load_16x32_layout(i, j):
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
    thread_id = i * 2 + j // 16
    row = thread_id % 16
    col = (j % 16) + (thread_id // 16) * 16
    return row, col

def permutation(n, h, w, c):
    i = n * output_height * output_width + (h - (c // in_channels) // kernel_w) * output_width + (w - (c // in_channels) % kernel_w)
    j = c + (h // output_height) * in_channels * kernel_w + (w // output_width) * in_channels
    print(n, h, w, c, i, j)
    kernel_i, kernel_j = A_global_16x32_to_shared_load_16x32_layout(i % wmma_m, j % wmma_k)
    row = (i // wmma_m) * 16 + kernel_i
    col = (j // wmma_k) * 32 + kernel_j
    return row, col

for n in range(0, 1):
    for h in range(0, 1):
        for w in range(1, 2):
            for c in range(0, 16):
                row, col = permutation(n, h, w, c)
                # print("row: ", row, "col: ", col)