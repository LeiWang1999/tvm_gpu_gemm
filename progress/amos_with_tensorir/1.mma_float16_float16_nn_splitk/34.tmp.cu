#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

#endif

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
extern "C" __global__ void __launch_bounds__(128) main_kernel2(half* __restrict__ A_global, half* __restrict__ B_global, half* __restrict__ C) {
  half TC_warp[256];
  __shared__ half A_global_shared[4096];
  __shared__ half B_global_shared[8192];
  half A_global_shared_warp[64];
  half B_global_shared_warp[32];

  const int MAX_BLOCK_N = 16;
  const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
  const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
  const auto totalBlock = gridDim.x * gridDim.y;
  const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
  const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
  const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
  const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
  const auto bz = blockIdx.z;
  const dim3 blockIdx(bx, by, bz);
  
  for (int i_0_2_init = 0; i_0_2_init < 8; ++i_0_2_init) {
    for (int j_0_2_init = 0; j_0_2_init < 4; ++j_0_2_init) {
      for (int i = 0; i < 8; ++i) {
TC_warp[((i_0_2_init * 32) + (j_0_2_init * 8)) + i] = 0.0;}
;
    }
  }
  for (int k_0_0 = 0; k_0_0 < 512; ++k_0_0) {
    __syncthreads();
    for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2 < 4; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2) {
      *(uint4*)(A_global_shared + (((((int)threadIdx.z) * 1024) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2 * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(A_global + (((((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.z) * 524288)) + ((ax0_0_ax1_0_ax0_1_ax1_1_fused_2 >> 1) * 262144)) + (k_0_0 * 512)) + ((ax0_0_ax1_0_ax0_1_ax1_1_fused_2 & 1) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + ((((int)threadIdx.x) >> 4) * 8)));
    }
    for (int ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 = 0; ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 < 8; ++ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1) {
      *(uint4*)(B_global_shared + (((((int)threadIdx.z) * 2048) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B_global + (((((((k_0_0 * 32768) + ((((int)threadIdx.z) >> 1) * 16384)) + (((int)blockIdx.x) * 4096)) + ((((int)threadIdx.z) & 1) * 2048)) + (ax0_0_ax1_0_ax0_1_ax1_1_fused_2_1 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + ((((int)threadIdx.x) >> 4) * 8)));
    }
    __syncthreads();
    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      for (int ax0_0 = 0; ax0_0 < 8; ++ax0_0) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_global_shared[((ax0_0 * 512) + (k_0_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_global_shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(A_global_shared_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(A_global_shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(A_global_shared_warp + (ax0_0 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {

  {
    unsigned int addr;
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_global_shared[(((k_0_1 * 4096) + (((int)threadIdx.z) * 1024)) + (ax1_0 * 256))])) + (((int)threadIdx.x) * 8)))
    );
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_global_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(B_global_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(B_global_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(B_global_shared_warp + (ax1_0 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int i_0_2 = 0; i_0_2 < 8; ++i_0_2) {
        for (int j_0_2 = 0; j_0_2 < 4; ++j_0_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(TC_warp + ((i_0_2 * 32) + (j_0_2 * 8))))[0]), "=r"(((unsigned *)(TC_warp + ((i_0_2 * 32) + (j_0_2 * 8))))[1])
      : "r"(((unsigned *)(A_global_shared_warp + (i_0_2 * 8)))[0]), "r"(((unsigned *)(A_global_shared_warp + (i_0_2 * 8)))[1]), "r"(((unsigned *)(A_global_shared_warp + (i_0_2 * 8)))[2]), "r"(((unsigned *)(A_global_shared_warp + (i_0_2 * 8)))[3]), "r"(((unsigned *)(B_global_shared_warp + (j_0_2 * 8)))[0]), "r"(((unsigned *)(B_global_shared_warp + (j_0_2 * 8)))[1]), "r"(((unsigned *)(TC_warp + ((i_0_2 * 32) + (j_0_2 * 8))))[0]), "r"(((unsigned *)(TC_warp + ((i_0_2 * 32) + (j_0_2 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(TC_warp + (((i_0_2 * 32) + (j_0_2 * 8)) + 4)))[0]), "=r"(((unsigned *)(TC_warp + (((i_0_2 * 32) + (j_0_2 * 8)) + 4)))[1])
      : "r"(((unsigned *)(A_global_shared_warp + (i_0_2 * 8)))[0]), "r"(((unsigned *)(A_global_shared_warp + (i_0_2 * 8)))[1]), "r"(((unsigned *)(A_global_shared_warp + (i_0_2 * 8)))[2]), "r"(((unsigned *)(A_global_shared_warp + (i_0_2 * 8)))[3]), "r"(((unsigned *)(B_global_shared_warp + ((j_0_2 * 8) + 4)))[0]), "r"(((unsigned *)(B_global_shared_warp + ((j_0_2 * 8) + 4)))[1]), "r"(((unsigned *)(TC_warp + (((i_0_2 * 32) + (j_0_2 * 8)) + 4)))[0]), "r"(((unsigned *)(TC_warp + (((i_0_2 * 32) + (j_0_2 * 8)) + 4)))[1]));
  }
        }
      }
    }
  }
  for (int ax0_0_1 = 0; ax0_0_1 < 8; ++ax0_0_1) {
    __syncthreads();
    for (int ax1_0_1 = 0; ax1_0_1 < 4; ++ax1_0_1) {
      for (int local_id = 0; local_id < 8; local_id+=2) {
*((uint *)&(&(A_global_shared[((((int)threadIdx.z) * 64) + (ax1_0_1 * 16))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 256) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))]) = *((uint *)&TC_warp[((ax0_0_1 * 32) + (ax1_0_1 * 8)) + local_id]);
}
;
    }
    __syncthreads();
    for (int ax0_ax1_fused_1 = 0; ax0_ax1_fused_1 < 32; ++ax0_ax1_fused_1) {
      atomicAdd((&(C[(((((((((int)blockIdx.y) * 131072) + (ax0_0_1 * 16384)) + ((ax0_ax1_fused_1 >> 1) * 1024)) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.z) * 64)) + ((ax0_ax1_fused_1 & 1) * 32)) + ((int)threadIdx.x))])), A_global_shared[(((((ax0_ax1_fused_1 >> 1) * 256) + (((int)threadIdx.z) * 64)) + ((ax0_ax1_fused_1 & 1) * 32)) + ((int)threadIdx.x))]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(8) main_kernel0(half* __restrict__ B_global, half* __restrict__ B) {
  *(uint4*)(B_global + ((((((((((int)blockIdx.y) >> 3) * 16384) + ((((int)blockIdx.x) & 15) * 1024)) + ((((int)threadIdx.x) >> 1) * 256)) + ((((int)blockIdx.y) & 3) * 64)) + ((((int)blockIdx.x) >> 4) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((((int)blockIdx.y) & 7) >> 2) * 8))) = *(uint4*)(B + (((((int)blockIdx.y) * 2048) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)));
}

extern "C" __global__ void __launch_bounds__(8) main_kernel1(half* __restrict__ A_global, half* __restrict__ A) {
  *(uint4*)(A_global + ((((((((int)blockIdx.y) >> 7) * 262144) + (((((((int)blockIdx.y) * 128) + (((int)blockIdx.x) * 4)) + (((int)threadIdx.x) >> 1)) & 1023) * 256)) + (((((int)blockIdx.y) & 63) >> 3) * 32)) + ((((int)threadIdx.x) & 1) * 16)) + (((((int)blockIdx.y) & 127) >> 6) * 8))) = *(uint4*)(A + (((((int)blockIdx.y) * 2048) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)));
}

