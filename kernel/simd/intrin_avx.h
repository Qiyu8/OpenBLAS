#define V_SIMD 256
#define V_SIMD_F64 1
/*
Data Type
*/
typedef __m256  v_f32;
typedef __m256d v_f64;
#define v_nlanes_f32 8
#define v_nlanes_f64 4
/*
arithmetic
*/

#define v_mul_f32 _mm256_mul_ps
#define v_mul_f64 _mm256_mul_pd
// Horizontal add: Calculates the sum of all vector elements.
BLAS_FINLINE float v_sum_f32(__m256 a)
{
    __m256 sum_halves = _mm256_hadd_ps(a, a);
    sum_halves = _mm256_hadd_ps(sum_halves, sum_halves);
    __m128 lo = _mm256_castps256_ps128(sum_halves);
    __m128 hi = _mm256_extractf128_ps(sum_halves, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    return _mm_cvtss_f32(sum);
}

BLAS_FINLINE double v_sum_f64(__m256d a)
{
    __m256d sum_halves = _mm256_hadd_pd(a, a);
    __m128d lo = _mm256_castpd256_pd128(sum_halves);
    __m128d hi = _mm256_extractf128_pd(sum_halves, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(sum);
}
/*
memory
*/
// unaligned load
#define v_load_f32 _mm256_loadu_ps
#define v_load_f64 _mm256_loadu_pd
BLAS_FINLINE __m256d v__setr_pd(double i0, double i1, double i2, double i3)
{
    return _mm256_setr_pd(i0, i1, i2, i3);
}
#define v_setf_f64(FILL, ...) v__setr_pd(V__SET_FILL_4(double, FILL, __VA_ARGS__))
#define v_set_f64(...) v_setf_f64(0, __VA_ARGS__)