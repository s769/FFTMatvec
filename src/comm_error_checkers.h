#ifndef __COMM_ERROR_CHECKERS_H__
#define __COMM_ERROR_CHECKERS_H__

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

#endif // __COMM_ERROR_CHECKERS_H__