
#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(1024) default_function_kernel0(float* __restrict__ C, float* __restrict__ A, float* __restrict__ B) {
  C[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int k = 0; k < 1024; ++k) {
    C[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (C[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (A[((k * 1024) + ((int)threadIdx.x))] * B[((k * 1024) + ((int)blockIdx.x))]));
  }
}

