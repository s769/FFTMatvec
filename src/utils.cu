#include "utils.hpp"

uint64_t Utils::getHostHash(const char* string)
{
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

/**
 * Returns the hostname of the machine.
 *
 * @param hostname The hostname to return.
 * @param maxlen The maximum length of the hostname.
 */

void Utils::getHostName(char* hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

__global__ void PadVectorKernel(const double2* const d_in, double2* const d_pad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size; j += blockDim.x) {
        if (j < size / 2)
            d_pad[(size_t)blockIdx.x * size + j] = d_in[(size_t)blockIdx.x * size / 2 + j];
        else
            d_pad[(size_t)blockIdx.x * size + j] = { 0, 0 };
    }
}

__global__ void PadVectorKernel(const double* const d_in, double* const d_pad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size; j += blockDim.x) {
        if (j < size / 2)
            d_pad[(size_t)blockIdx.x * size + j] = d_in[(size_t)blockIdx.x * size / 2 + j];
        else
            d_pad[(size_t)blockIdx.x * size + j] = 0;
    }
}

__global__ void UnpadVectorKernel(const double2* const d_in, double2* const d_unpad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size / 2; j += blockDim.x) {
        d_unpad[(size_t)blockIdx.x * size / 2 + j] = d_in[(size_t)blockIdx.x * size + j];
    }
}
__global__ void UnpadVectorKernel(const double* const d_in, double* const d_unpad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size / 2; j += blockDim.x) {
        d_unpad[(size_t)blockIdx.x * size / 2 + j] = d_in[(size_t)blockIdx.x * size + j];
    }
}

__global__ void RepadVectorKernel(const double2* const d_in, double2* const d_unpad,
    const unsigned int num_cols, const unsigned int size)
{
    int t = threadIdx.x;
    for (int j = t; j < size; j += blockDim.x) {
        if (j < size / 2)
            d_unpad[(size_t)blockIdx.x * size + j] = d_in[(size_t)blockIdx.x * size + j];
        else if (size % 2 == 1 && j == size / 2)
            d_unpad[(size_t)blockIdx.x * size + j] = { d_in[(size_t)blockIdx.x * size + j].x, 0 };
        else
            d_unpad[(size_t)blockIdx.x * size + j] = { 0, 0 };
    }
}

void Utils::PadVector(const double* const d_in, double* const d_pad, const unsigned int num_cols,
    const unsigned int size, cudaStream_t s)
{
    if (size % 4 == 0)
        PadVectorKernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
            reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_pad), num_cols,
            size / 2);
    else
        PadVectorKernel<<<num_cols, std::min((int)(size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(
            d_in, d_pad, num_cols, size);
    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}

void Utils::UnpadRepadVector(const double* const d_in, double* const d_out,
    const unsigned int num_cols, const unsigned int size, const bool unpad, cudaStream_t s)
{
    if (unpad) {
        if (size % 4 == 0)
            UnpadVectorKernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
                reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_out), num_cols,
                size / 2);
        else
            UnpadVectorKernel<<<num_cols, std::min((int)(size + 1) / 2, MAX_BLOCK_SIZE), 0, s>>>(
                d_in, d_out, num_cols, size);
    } else {
        RepadVectorKernel<<<num_cols, std::min((int)(size + 3) / 4, MAX_BLOCK_SIZE), 0, s>>>(
            reinterpret_cast<const double2*>(d_in), reinterpret_cast<double2*>(d_out), num_cols,
            size / 2);
    }
    gpuErrchk(cudaPeekAtLastError());
#if ERR_CHK

    gpuErrchk(cudaDeviceSynchronize());
#endif
}

void Utils::printVec(double* vec, int len, int unpad_size, std::string name)
{
    double* h_vec;
    h_vec = (double*)malloc(len * unpad_size * sizeof(double));
    gpuErrchk(cudaMemcpy(h_vec, vec, len * unpad_size * sizeof(double), cudaMemcpyDeviceToHost));
    printf("%s:\n", name.c_str());

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < unpad_size; j++) {
            printf("block: %d, t: %d, val: %f\n", i, j, h_vec[i * unpad_size + j]);
        }
        printf("\n");
    }
    free(h_vec);
}

void Utils::printVecMPI(
    double* vec, int len, int unpad_size, int rank, int world_size, std::string name)
{
    if (rank == 0) {
        printf("%s:\n", name.c_str());
    }
    for (int r = 0; r < world_size; r++) {
        if (rank == r) {
            printf("Rank: %d\n", r);
            printVec(vec, len, unpad_size);
        }
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
}

void Utils::printVecComplex(Complex* vec, int len, int unpad_size, std::string name)
{

    Complex* h_vec;
    h_vec = (Complex*)malloc(len * unpad_size * sizeof(Complex));
    gpuErrchk(cudaMemcpy(h_vec, vec, len * unpad_size * sizeof(Complex), cudaMemcpyDeviceToHost));

    printf("%s:\n", name.c_str());

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < unpad_size; j++) {
            printf("block: %d, t: %d, val: %f + %f i\n", i, j, h_vec[i * unpad_size + j].x,
                h_vec[i * unpad_size + j].y);
        }
        printf("\n");
    }
    free(h_vec);
}

void Utils::makeTable(std::vector<std::string> col_names, std::vector<long double> mean,
    std::vector<long double> min, std::vector<long double> max)
{

    int size = col_names.size();

    if (mean.size() != size - 1 || min.size() != size - 1 || max.size() != size - 1) {
        std::cerr << "Error: makeTable: input vectors must have the same size" << std::endl;
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        return;
    }

    tabulate::Table table;
    // table.add_row({title});
    table.format().font_align(tabulate::FontAlign::center);
    table.format().font_style({ tabulate::FontStyle::bold });

    tabulate::Table::Row_t col_names_row(col_names.begin(), col_names.end());
    table.add_row(col_names_row);
    table[0][0]
        .format()
        .font_color(tabulate::Color::yellow)
        .font_style({ tabulate::FontStyle::bold, tabulate::FontStyle::underline });

    // convert long double vectors to string vectors and add the row titles first
    std::vector<std::string> mean_str, min_str, max_str;
    mean_str.push_back("Mean Time (s)");
    min_str.push_back("Min Time (s)");
    max_str.push_back("Max Time (s)");

    for (int i = 0; i < size - 1; i++) {
        mean_str.push_back(std::to_string(mean[i]));
        min_str.push_back(std::to_string(min[i]));
        max_str.push_back(std::to_string(max[i]));
    }

    tabulate::Table::Row_t min_row(min_str.begin(), min_str.end());
    tabulate::Table::Row_t mean_row(mean_str.begin(), mean_str.end());
    tabulate::Table::Row_t max_row(max_str.begin(), max_str.end());

    table.add_row(mean_row);
    table.add_row(min_row);
    table.add_row(max_row);

    std::cout << table << std::endl;
}

void Utils::printRaw(long double* mean_times, long double* min_times, long double* max_times,
    long double* mean_times_f, long double* min_times_f, long double* max_times_f,
    long double* mean_times_fs, long double* min_times_fs, long double* max_times_fs,
    int world_size, int times_len)
{

    for (int i = 0; i < 3; i++)
        printf("%Lf\t", min_times[i]);
    printf("\n");
    for (int i = 0; i < 3; i++)
        printf("%Lf\t", mean_times[i]);
    printf("\n");
    for (int i = 0; i < 3; i++)
        printf("%Lf\t", max_times[i]);
    printf("\n");
    for (int i = 0; i < times_len; i++)
        printf("%Lf\t", min_times_f[i]);
    printf("\n");
    for (int i = 0; i < times_len; i++)
        printf("%Lf\t", mean_times_f[i]);
    printf("\n");
    for (int i = 0; i < times_len; i++)
        printf("%Lf\t", max_times_f[i]);
    printf("\n");
    for (int i = 0; i < times_len; i++)
        printf("%Lf\t", min_times_fs[i]);
    printf("\n");
    for (int i = 0; i < times_len; i++)
        printf("%Lf\t", mean_times_fs[i]);
    printf("\n");
    for (int i = 0; i < times_len; i++)
        printf("%Lf\t", max_times_fs[i]);
    printf("\n");
}

void Utils::printTimes(int reps, bool table)
{
#if TIME_MPI

    int world_rank, world_size;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    int times_len = t_list_f.size();
    long double mpi_times_f[times_len], mpi_times_fs[times_len], mpi_times[3];
    long double *mean_times, *min_times, *max_times, *mean_times_f, *min_times_f, *max_times_f,
        *mean_times_fs, *min_times_fs, *max_times_fs;

    if (world_rank == 0) {
        mean_times = new long double[3];
        min_times = new long double[3];
        max_times = new long double[3];
        mean_times_f = new long double[times_len];
        min_times_f = new long double[times_len];
        max_times_f = new long double[times_len];
        mean_times_fs = new long double[times_len];
        min_times_fs = new long double[times_len];
        max_times_fs = new long double[times_len];
    }

    for (int i = 0; i < times_len; i++) {
        mpi_times_f[i] = t_list_f[i].seconds / reps;
        mpi_times_fs[i] = t_list_fs[i].seconds / reps;
    }

    mpi_times[0] = t_list[ProfilerTimesFull::COMM_INIT].seconds;
    mpi_times[1] = t_list[ProfilerTimesFull::SETUP].seconds;
    mpi_times[2] = t_list[ProfilerTimesFull::FULL].seconds;

    MPICHECK(MPI_Reduce(
        (void*)&mpi_times, (void*)mean_times, 3, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(
        (void*)&mpi_times, (void*)min_times, 3, MPI_LONG_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(
        (void*)&mpi_times, (void*)max_times, 3, MPI_LONG_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void*)&mpi_times_f, (void*)mean_times_f, times_len, MPI_LONG_DOUBLE,
        MPI_SUM, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void*)&mpi_times_f, (void*)min_times_f, times_len, MPI_LONG_DOUBLE,
        MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void*)&mpi_times_f, (void*)max_times_f, times_len, MPI_LONG_DOUBLE,
        MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void*)&mpi_times_fs, (void*)mean_times_fs, times_len, MPI_LONG_DOUBLE,
        MPI_SUM, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void*)&mpi_times_fs, (void*)min_times_fs, times_len, MPI_LONG_DOUBLE,
        MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void*)&mpi_times_fs, (void*)max_times_fs, times_len, MPI_LONG_DOUBLE,
        MPI_MAX, 0, MPI_COMM_WORLD));

    for (int i = 0; i < 3; i++) {
        mean_times[i] /= world_size;
    }

    for (int i = 0; i < times_len; i++) {
        mean_times_f[i] /= world_size;
        mean_times_fs[i] /= world_size;
    }

    if (world_rank == 0) {
        if (table) {
            std::cout << "Aggregate Times: " << std::endl;
            std::cout << std::endl;

            makeTable({ "Summary", "Initialize", "Setup", "Matvecs" },
                { mean_times[0], mean_times[1], mean_times[2] },
                { min_times[0], min_times[1], min_times[2] },
                { max_times[0], max_times[1], max_times[2] });
            std::vector<long double> mean_times_v(mean_times_f, mean_times_f + times_len);
            std::vector<long double> min_times_v(min_times_f, min_times_f + times_len);
            std::vector<long double> max_times_v(max_times_f, max_times_f + times_len);

            std::cout << std::endl;

            std::cout << "Average Times per Matvec: " << std::endl;

            std::cout << std::endl;

            makeTable({ "F Matvec", "Broadcast", "Pad", "FFT", "SOTI-to-TOSI", "SBGEMV",
                          "TOSI-to-SOTI", "IFFT", "Unpad", "Reduce", "Total" },
                mean_times_v, min_times_v, max_times_v);

            std::cout << std::endl;

            mean_times_v = std::vector<long double>(mean_times_fs, mean_times_fs + times_len);
            min_times_v = std::vector<long double>(min_times_fs, min_times_fs + times_len);
            max_times_v = std::vector<long double>(max_times_fs, max_times_fs + times_len);

            makeTable({ "F* Matvec", "Broadcast", "Pad", "FFT", "SOTI-to-TOSI", "SBGEMV",
                          "TOSI-to-SOTI", "IFFT", "Unpad", "Reduce", "Total" },
                mean_times_v, min_times_v, max_times_v);
        } else {
            printRaw(mean_times, min_times, max_times, mean_times_f, min_times_f, max_times_f,
                mean_times_fs, min_times_fs, max_times_fs, world_size, times_len);
        }

        delete[] mean_times;
        delete[] min_times;
        delete[] max_times;
        delete[] mean_times_f;
        delete[] min_times_f;
        delete[] max_times_f;
        delete[] mean_times_fs;
        delete[] min_times_fs;
        delete[] max_times_fs;
    }

#endif
}
