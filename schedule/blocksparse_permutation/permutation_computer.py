# do permutation
# 32x64 -> 64 x 32
def permutation_func(i, j):
    lane_id = (i*64 + j) // 4
    element_id = (i*64 + j) % 4
    res_i = element_id + (j // 4) * 4
    res_j = (lane_id + lane_id // 32) % 32
    return res_i, res_j

for m in range(0, 32):
    for n in range(0, 64):
        print(m, n, permutation_func(m, n))
