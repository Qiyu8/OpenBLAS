#define V_SIMD 128
#ifdef __aarch64__
    #define V_SIMD_F64 1
#else
    #define V_SIMD_F64 0
#endif
/*
Data Type
*/
typedef float32x4_t v_f32;
#if V_SIMD_F64
typedef float64x2_t v_f64;
#endif
#define v_nlanes_f32 4
#define v_nlanes_f64 2
/*
arithmetic
*/
#define v_mul_f32 vmulq_f32
#define v_mul_f64 vmulq_f64

// Horizontal add: Calculates the sum of all vector elements.
BLAS_FINLINE float v_sum_f32(float32x4_t a)
{
    float32x2_t r = vadd_f32(vget_high_f32(a), vget_low_f32(a));
    return vget_lane_f32(vpadd_f32(r, r), 0);
}
#ifdef __aarch64__
BLAS_FINLINE double v_sum_f64(float64x2_t a)
{
    return vget_lane_f64(vget_low_f64(a) + vget_high_f64(a), 0);
}
#endif
/*
memory
*/
// unaligned load
#define v_load_f32(a) vld1q_f32((const float*)a)
#define v_load_f64(a) vld1q_f64((const double*)a)
