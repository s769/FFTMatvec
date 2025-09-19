#ifndef __COMM_ERROR_CHECKERS_H__
#define __COMM_ERROR_CHECKERS_H__

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) // Macros defined by hipcc
#include <rccl.h>                                                     // For HIP compilation
#else
#include <nccl.h> // For CUDA compilation
#endif

#define MPICHECK(cmd)                                \
    do                                               \
    {                                                \
        int e = cmd;                                 \
        if (e != MPI_SUCCESS)                        \
        {                                            \
            printf("Failed: MPI error %s:%d '%d'\n", \
                   __FILE__, __LINE__, e);           \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while (0)

#define NCCLCHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        ncclResult_t r = cmd;                                  \
        if (r != ncclSuccess)                                  \
        {                                                      \
            printf("Failed, NCCL error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#endif // __COMM_ERROR_CHECKERS_H__