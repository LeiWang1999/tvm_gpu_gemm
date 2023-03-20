# do permutation
def A_permutation(m, n):
    # 8x8
    # to 8x8
    # 0,0 -> 0, 0
    # 1,0 -> 1, 1
    # 2,0 -> 2, 2
    # 0,1 -> 0, 1
    # 1,1 -> 1, 2
    # 2,1 -> 2, 3
    res_m = m
    res_n = (n + m) % 8
    return res_m, res_n

for m in range(0, 8):
    for n in range(0, 8):
        print(m, n, A_permutation(m, n))
