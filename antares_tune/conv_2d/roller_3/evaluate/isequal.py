import random

def cuiwei(x):
    return (x // 42) % 42

def ours(x):
    return (x % 1764) // 42

# gen randomint and test if the output equal of cuiwei and ours
for i in range(1000):
    x = random.randint(0, 1e5)
    # print(x, cuiwei(x), ours(x), cuiwei(x) == ours(x))
    assert cuiwei(x) == ours(x)