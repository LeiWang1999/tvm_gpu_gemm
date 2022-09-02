-technical post-

https://zhuanlan.zhihu.com/p/560729749

-info-

shape : 16384, 16384, 16384

dtype: float32

device: gpu rtx 3090 24GB

cuda version : 11.1

tvm version: 10.0

cutlass baseline: 421 ms

cublas baseline: 420 ms

-result-

0.native_gemm.py：

average time cost of 10 runs = 2.15598 ms, 996.059 GFLOPS

1.blocked_gemm.py：

average time cost of 10 runs = 4865.28 ms, 1807.93 GFLOPS.

2.thread_tiling.py：

average time cost of 10 runs = 854.843 ms, 10289.7 GFLOPS.

3.wrap_tiling.py：

average time cost of 10 runs = 565.302 ms, 15560 GFLOPS.

4.vectorize.py：

average time cost of 10 runs = 423.255 ms, 20782 GFLOPS.

5.double_buffer.py：

average time cost of 10 runs = 435.032 ms, 20219.4 GFLOPS.

