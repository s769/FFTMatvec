
/**
 * @file shared.hpp
 * @brief Header file containing shared definitions and includes.
 *
 * This file contains various definitions and includes that are commonly used across multiple source files.
 * It includes standard C and C++ libraries, CUDA libraries, OpenMP, MPI, NCCL, and other necessary headers.
 * It also defines various constants, types, and helper functions for error checking.
 *
 * @note This file should be included in all source files that require these shared definitions and includes.
 */
#ifndef __SHARED_H__
#define __SHARED_H__

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <cutensor.h>
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5PropertyList.hpp>
#include <highfive/H5Easy.hpp>

#include <omp.h>


#include <cublas_v2.h>



#define TIME_MPI 1
#define ERR_CHK 1
#define FFT_64 0
#define ROW_SETUP 1

#if FFT_64
typedef long long int fft_int_t;
#else
typedef int fft_int_t;
#endif


#include <string>
#include <vector>


#include <mpi.h>
#include <unistd.h>
#include <nccl.h>

#if TIME_MPI
#include "profiler.hpp"
#include <array>
#endif

typedef double2 Complex;
typedef Complex data_t;


#include "error_checkers.h"
#include "comm_error_checkers.h"



#if TIME_MPI
template <typename E, class T, std::size_t N>
class enum_array : public std::array<T, N>
{
public:
    T &operator[](E e)
    {
        return std::array<T, N>::operator[]((std::size_t)e);
    }

    const T &operator[](E e) const
    {
        return std::array<T, N>::operator[]((std::size_t)e);
    }

    T &operator[](int e)
    {
        return std::array<T, N>::operator[]((std::size_t)e);
    }

    const T &operator[](int e) const
    {
        return std::array<T, N>::operator[]((std::size_t)e);
    }
};

enum class ProfilerTimesFull : unsigned int
{
    COMM_INIT = 0,
    SETUP,
    FULL,
};


enum class ProfilerTimes : unsigned int
{
    BROADCAST = 0,
    PAD,
    FFT,
    TRANS1,
    SBGEMV,
    TRANS2,
    IFFT,
    UNPAD,
    NCCLC,
    TOT,
};



extern enum_array<ProfilerTimesFull, profiler_t, 3> t_list;
extern enum_array<ProfilerTimes, profiler_t, 10> t_list_f;
extern enum_array<ProfilerTimes, profiler_t, 10> t_list_fs;


#endif

#endif // __SHARED_H__