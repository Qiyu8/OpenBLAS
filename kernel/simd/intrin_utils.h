#ifndef _INTRIN_UTILS_H
#define _INTRIN_UTILS_H
#define EXPAND(x) x
#if defined(__GNUC__) || defined(__ICC) || defined(__clang__)
    #define DECL_ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined(_MSC_VER)
    #define DECL_ALIGNED(x) __declspec(align(x))
#else
    #define DECL_ALIGNED(x)
#endif
#define V__SET_2(CAST, I0, I1, ...) (CAST)(I0), (CAST)(I1)

#define V__SET_4(CAST, I0, I1, I2, I3, ...) \
    (CAST)(I0), (CAST)(I1), (CAST)(I2), (CAST)(I3)

#define V__SET_8(CAST, I0, I1, I2, I3, I4, I5, I6, I7, ...) \
    (CAST)(I0), (CAST)(I1), (CAST)(I2), (CAST)(I3), (CAST)(I4), (CAST)(I5), (CAST)(I6), (CAST)(I7)

#define V__SET_FILL_2(CAST, F, ...) EXPAND(V__SET_2(CAST, __VA_ARGS__, F, F))

#define V__SET_FILL_4(CAST, F, ...) EXPAND(V__SET_4(CAST, __VA_ARGS__, F, F, F, F))

#define V__SET_FILL_8(CAST, F, ...) EXPAND(V__SET_8(CAST, __VA_ARGS__, F, F, F, F, F, F, F, F))


#endif