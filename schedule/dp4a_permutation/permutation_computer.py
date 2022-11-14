# do permutation
def B_local_permutation(m, n):
    # 32 * 128
    start_pos_m = (m // 4) * 4
    start_pos_n = (n // 4) * 4
    temp_m = n % 4
    temp_n = m % 4
    return (start_pos_m + temp_m, start_pos_n + temp_n)

for m in range(0, 16):
    for n in range(0, 16):
        print(m, n, B_local_permutation(m, n))
