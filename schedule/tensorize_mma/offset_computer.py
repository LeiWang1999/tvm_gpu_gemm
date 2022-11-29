k_0 = 0
by = 0
bx = 0
for ty in range(0, 1):
    for tx in range(0, 16):
        print("ty: ", ty, "tx: ", tx)
        idx = ((((tx) & 15) * 16) + (((tx) >> 4) * 8))
        print("A_local_shared Index: {0}, [{1},{2}]".format(idx, idx // 8, idx % 8))
        
