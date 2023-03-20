# do permutation
# 32x64 -> 64 x 32
def shared_16x16_to_ldmatrix_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)

for m in range(0, 16):
    for n in range(0, 16):
        print(m, n, shared_16x16_to_ldmatrix_32x8_layout(m, n))
