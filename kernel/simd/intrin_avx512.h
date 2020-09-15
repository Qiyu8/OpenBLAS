#define V_SIMD 512
#define V_SIMD_F64 1
/*
Data Type
*/
typedef __m512  v_f32;
typedef __m512d v_f64;
#define v_nlanes_f32 16
#define v_nlanes_f64 8
/*
arithmetic
*/
#define v_mul_f32 _mm512_mul_ps
#define v_mul_f64 _mm512_mul_pd

#ifdef HAVE_AVX512F_REDUCE
#define v_sum_f32 _mm512_reduce_add_ps
#define v_sum_f64 _mm512_reduce_add_pd
#else
BLAS_FINLINE float v_sum_f32(v_f32 a)
{
    __m512 h64 = _mm512_shuffle_f32x4(a, a, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 sum32 = _mm512_add_ps(a, h64);
    __m512 h32 = _mm512_shuffle_f32x4(sum32, sum32, _MM_SHUFFLE(1, 0, 3, 2));
    __m512 sum16 = _mm512_add_ps(sum32, h32);
    __m512 h16 = _mm512_permute_ps(sum16, _MM_SHUFFLE(1, 0, 3, 2));
    __m512 sum8 = _mm512_add_ps(sum16, h16);
    __m512 h4 = _mm512_permute_ps(sum8, _MM_SHUFFLE(2, 3, 0, 1));
    __m512 sum4 = _mm512_add_ps(sum8, h4);
    return _mm_cvtss_f32(_mm512_castps512_ps128(sum4));
}
BLAS_FINLINE double v_sum_f64(v_f64 a)
{
    __m512d h64 = _mm512_shuffle_f64x2(a, a, _MM_SHUFFLE(3, 2, 3, 2));
    __m512d sum32 = _mm512_add_pd(a, h64);
    __m512d h32 = _mm512_permutex_pd(sum32, _MM_SHUFFLE(1, 0, 3, 2));
    __m512d sum16 = _mm512_add_pd(sum32, h32);
    __m512d h16 = _mm512_permute_pd(sum16, _MM_SHUFFLE(2, 3, 0, 1));
    __m512d sum8 = _mm512_add_pd(sum16, h16);
    return _mm_cvtsd_f64(_mm512_castpd512_pd128(sum8));
}
#endif
/*
memory
*/
// unaligned load
#define v_load_f32(PTR) _mm512_loadu_ps((const __m512*)(PTR))
#define v_load_f64(PTR) _mm512_loadu_pd((const __m512d*)(PTR))