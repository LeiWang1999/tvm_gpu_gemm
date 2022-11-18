k_0 = 0
by = 0
bx = 0
for ty in range(0, 16):
    for tx in range(0, 16):
        print("ty: ", ty, "tx: ", tx)
        for ax1_1_1 in range(0, 4):
            idx = ((((((ty) >> 1) * 128) + (ax1_1_1 * 32)) +
                    (((ty) & 1) * 16)) + (tx))
            print("B_local_shared Index: {0}, [{1},{2}]".format(idx, idx // 32, idx % 32),  end=", ")
            idx = ((ax1_1_1 * 4))
            print("B_local_local Index{0}, [{1}, {2}]".format(idx, idx // 4, idx % 4))
