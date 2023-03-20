def access_pattern(bz=0, bx=0, by=0, tz=0, ty=0, tx=0):
    return  ((tx >> 3) * 8) + ((tx >> 3) + (tx& 7) & 7)

for tx in range(0, 32):
    print("threadIdx: ", tx, access_pattern(tx=tx))
