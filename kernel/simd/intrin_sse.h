#define V_SIMD 128
#define V_SIMD_F64 1
/*
Data Type
*/
typedef __m128  v_f32;
typedef __m128d v_f64;
#define v_nlanes_f32 4
#define v_nlanes_f64 2
/*
arithmetic
*/

#define v_mul_f32 _mm_mul_ps
#define v_mul_f64 _mm_mul_pd
// Horizontal add: Calculates the sum of all vector elements.
BLAS_FINLINE float v_sum_f32(__m128 a)
{
#ifdef HAVE_SSE3
    __m128 sum_halves = _mm_hadd_ps(a, a);
    return _mm_cvtss_f32(_mm_hadd_ps(sum_halves, sum_halves));
#else
    __m128 t1 = _mm_movehl_ps(a, a);
    __m128 t2 = _mm_add_ps(a, t1);
    __m128 t3 = _mm_shuffle_ps(t2, t2, 1);
    __m128 t4 = _mm_add_ss(t2, t3);
    return _mm_cvtss_f32(t4);
#endif
}

BLAS_FINLINE double v_sum_f64(__m128d a)
{
#ifdef HAVE_SSE3
    return _mm_cvtsd_f64(_mm_hadd_pd(a, a));
#else
    return _mm_cvtsd_f64(_mm_add_pd(a, _mm_unpackhi_pd(a, a)));
#endif
}

/*
memory
*/
// unaligned load
#define v_load_f32 _mm_loadu_ps
#define v_load_f64 _mm_loadu_pd
