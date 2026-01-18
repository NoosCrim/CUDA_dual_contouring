#include <cinttypes>
#include <concepts>
#pragma once

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

inline constexpr unsigned int opt_align(unsigned int size)
{
    if(size>>4)
        return 16;
    if(size>>3)
        return 8;
    if(size>>2)
        return 4;
    if(size>>1)
        return 2;
    return 1;
}

template<typename T, unsigned int N>
struct alignas( opt_align( opt_align( sizeof(T) ) * N)) vec_t
{
private:
    alignas(opt_align(sizeof(T))) T _v[N];
public:
    constexpr inline __host__ __device__ vec_t(): _v{} {}
    template<typename... ARGS>
    constexpr inline __host__ __device__ vec_t(ARGS... args): _v{(T)args...} {}
    template<typename T2, unsigned int N2>
    constexpr inline __host__ __device__ vec_t(const vec_t<T2, N2> &other)
    {
        #pragma unroll
        for(int i = 0; i < N && i < N2; i++)
            _v[i] = (T)other[i];
        
        #pragma unroll
        for(int i = N2; i < N; i++)
            _v[i] = 0;
    }

    constexpr inline __host__ __device__ T& operator[](unsigned int i)
    {
        return _v[i];
    }

    constexpr inline __host__ __device__ T operator[](unsigned int i) const
    {
        return _v[i];
    }

    constexpr inline __host__ __device__ vec_t<T, N> operator-() const
    {
        vec_t<T,N> out;
        #pragma unroll
        for(int i = 0; i < N; i++)
            out[i] = -_v[i];
        return out;
    }

    template<typename T2, unsigned int N2>
    constexpr inline __host__ __device__ vec_t<T, N>& operator=(const vec_t<T2, N2> &other)
    {
        #pragma unroll
        for(int i = 0; i < N & i < N2; i++)
            _v[i] = (T)other[i];
        
        return *this;
    }

    constexpr inline __host__ __device__ T lenSqr()
    {
        T out{};
        for(int i = 0; i < N; i++)
            out += _v[i] * _v[i];
        return out;
    }
    
    constexpr inline __host__ __device__ T len()
    {
        return sqrt(lenSqr());
    }

    constexpr inline __host__ __device__ vec_t<T,N> norm()
    {
        return *this/len();
    }
};

template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N>& operator+=(vec_t<T,N> &a, const vec_t<T2,N> &b)
{
    #pragma unroll
    for(int i = 0; i < N; i++)
        a[i] = a[i] + b[i];
    return a;
}
template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N>& operator-=(vec_t<T,N> &a, const vec_t<T2,N> &b)
{
    #pragma unroll
    for(int i = 0; i < N; i++)
        a[i] = a[i] - b[i];
    return a;
}
template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N>& operator*=(vec_t<T,N> &a, const vec_t<T2,N> &b)
{
    #pragma unroll
    for(int i = 0; i < N; i++)
        a[i] = a[i] * b[i];
    return a;
}
template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N>& operator/=(vec_t<T,N> &a, const vec_t<T2,N> &b)
{
    #pragma unroll
    for(int i = 0; i < N; i++)
        a[i] = a[i] / b[i];
    return a;
}
template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N>& operator*=(vec_t<T,N> &v, T2 s)
{
    #pragma unroll
    for(int i = 0; i < N; i++)
        v[i] = v[i] * s;
    return v;
}
template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N>& operator/=(vec_t<T,N> &v, T2 s)
{
    #pragma unroll
    for(int i = 0; i < N; i++)
        v[i] = v[i] / s;
    return v;
}

template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N> operator+(const vec_t<T,N> &a, const vec_t<T2,N> &b)
{
    vec_t<T,N> out;
    #pragma unroll
    for(int i = 0; i < N; i++)
        out[i] = a[i] + b[i];
    return out;
}
template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N> operator-(const vec_t<T,N> &a, const vec_t<T2,N> &b)
{
    vec_t<T,N> out;
    #pragma unroll
    for(int i = 0; i < N; i++)
        out[i] = a[i] - b[i];
    return out;
}
template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N> operator*(const vec_t<T,N> &a, const vec_t<T2,N> &b)
{
    vec_t<T,N> out;
    #pragma unroll
    for(int i = 0; i < N; i++)
        out[i] = a[i] * b[i];
    return out;
}
template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N> operator/(const vec_t<T,N> &a, const vec_t<T2,N> &b)
{
    vec_t<T,N> out;
    #pragma unroll
    for(int i = 0; i < N; i++)
        out[i] = a[i] / b[i];
    return out;
}
template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N> operator*(const vec_t<T,N> &v, T2 s)
{
    vec_t<T,N> out;
    #pragma unroll
    for(int i = 0; i < N; i++)
        out[i] = v[i] * s;
    return out;
}
template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N> operator/(const vec_t<T,N> &v, T2 s)
{
    vec_t<T,N> out;
    #pragma unroll
    for(int i = 0; i < N; i++)
        out[i] = v[i] / s;
    return out;
}

template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ vec_t<T,N> operator*(T2 s, const vec_t<T2,N> &v)
{
    vec_t<T,N> out;
    #pragma unroll
    for(int i = 0; i < N; i++)
        out[i] = v[i] * s;
    return out;
}

template<typename T, typename T2, unsigned int N>
constexpr inline __host__ __device__ bool operator==(const vec_t<T,N> &a, const vec_t<T2,N> &b)
{
    bool out = true;
    #pragma unroll
    for(int i = 0; i < N; i++)
        out = out && (a[i] == b[i]);
    return out;
}

template<typename T>
constexpr inline __host__ __device__  vec_t<T, 3> cross(const vec_t<T, 3> &a, const vec_t<T, 3> &b)
{
    return { 
        a[1]*b[2] - a[2]*b[1], 
        a[2]*b[0] - a[0]*b[2], 
        a[0]*b[1] - a[1]*b[0]
    };
}

// expands 10-bit int into 30-bit with 2-bit gaps
// undefined if v >= 1024
constexpr inline __host__ __device__ uint32_t expand_bits(uint32_t v)
{
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

// squishes expanded bits back together
constexpr inline __host__ __device__ uint32_t squish_bits(uint32_t v)
{
    v &= 0x09249249;                  // v = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    v = (v | (v >> 2)) & 0x030C30C3; 
    v = (v | (v >> 4)) & 0x0300F00F; 
    v = (v | (v >> 8)) & 0xff0000ff; 
    v = (v | (v >> 16)) & 0x000003ff;
    return v;
}

// calculates morton code for 3D index
// undefined if any component >= 1024
constexpr inline __host__ __device__ uint32_t to_morton(vec_t<uint32_t, 3> i)
{
    return expand_bits(i[0]) | expand_bits(i[1]) << 1 | expand_bits(i[2]) << 2;
}

constexpr inline __host__ __device__ vec_t<uint32_t, 3> from_morton(uint32_t v)
{
    constexpr uint32_t bits = 0x49249249;
    return {squish_bits(v & bits), squish_bits(v>>1 & bits), squish_bits(v>>2 & bits)};
}

using density_t = float;
using density_compressed_t = uint8_t;
using vec3_t = vec_t<float,3>;
using gradient_t = vec3_t;

// stores hermite data as octahedron mapped gradient direction 
// and position as fixed 8 bit fraction encoding position across the edge
using hermite_compressed_t = vec_t<uint8_t, 3>;

constexpr inline density_compressed_t __host__ __device__ density_encode(density_t v)
{
    return v < 0.f ? 0 : (v > 1.f ? (density_compressed_t)-1 : ((density_compressed_t)-1)*v);
}

constexpr inline density_t __host__ __device__ density_decode(density_compressed_t v)
{
    return ((density_t)v) / (density_compressed_t)-1;
}

template<typename F>
concept DensityFunction = requires(F f, vec3_t p, density_t prev_v) {
    { f(p, prev_v) } -> std::same_as<density_t>;
};
template<typename F>
concept GradientFunction = requires(F f, vec3_t p, gradient_t prev_v) {
    { f(p, prev_v) } -> std::same_as<gradient_t>;
};