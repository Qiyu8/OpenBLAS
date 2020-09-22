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
typedef float64x2x2_t v_f64x2;
#endif
#define v_nlanes_f32 4
#define v_nlanes_f64 2
/*
arithmetic
*/
#define v_add_f32 vaddq_f32
#define v_add_f64 vaddq_f64
#define v_mul_f32 vmulq_f32
#define v_mul_f64 vmulq_f64

/***************************
 * FUSED F32
 ***************************/
#ifdef HAVE_VFPV4 // FMA
    // multiply and add, a*b + c
    BLAS_FINLINE v_f32 v_muladd_f32(v_f32 a, v_f32 b, v_f32 c)
    { return vfmaq_f32(c, a, b); }
#else
    // multiply and add, a*b + c
    BLAS_FINLINE v_f32 v_muladd_f32(v_f32 a, v_f32 b, v_f32 c)
    { return vmlaq_f32(c, a, b); }
#endif
/***************************
 * FUSED F64
 ***************************/
#if V_SIMD_F64
    BLAS_FINLINE v_f64 v_muladd_f64(v_f64 a, v_f64 b, v_f64 c)
    { return vfmaq_f64(c, a, b); }
#endif // V_SIMD_F64

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
#define v_loadu_f32(a) vld1q_f32((const float*)a)
#define v_loadu_f64(a) vld1q_f64((const double*)a)
#ifdef __aarch64__
BLAS_FINLINE float64x2_t v__set_f64(double i0, double i1)
{
    const double DECL_ALIGNED(16) data[2] = {i0, i1};
    return vld1q_f64(data);
}
#define v_setf_f64(FILL, ...) v__set_f64(V__SET_FILL_2(double, FILL, __VA_ARGS__))
#define v_set_f64(...) v_setf_f64(0, __VA_ARGS__)
#endif
#define v_zero_f32() vdupq_n_f32(0.0f)
#define v_zero_f64() vdupq_n_f64(0.0)
/*
convert
*/
#ifdef __aarch64__
BLAS_FINLINE v_f64x2 v_cvt_f64_f32(float32x4_t a) {
    v_f64x2 r;
    r.val[0] = vcvt_f64_f32(vget_low_f32(a));
    r.val[1] = vcvt_f64_f32(vget_high_f32(a)));
    return r;
}
#endif