#include "utils.hpp"
#include "util_kernels.hpp"

uint64_t Utils::get_host_hash(const char *string)
{
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++)
    {
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

void Utils::get_host_name(char *hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++)
    {
        if (hostname[i] == '.')
        {
            hostname[i] = '\0';
            return;
        }
    }
}

void Utils::print_vec(double *vec, int len, int block_size, std::string name)
{
    double *h_vec;
    h_vec = (double *)malloc((size_t)len * block_size * sizeof(double));
    gpuErrchk(
        cudaMemcpy(h_vec, vec, (size_t)len * block_size * sizeof(double), cudaMemcpyDeviceToHost));
    printf("%s:\n", name.c_str());

    for (size_t i = 0; i < len; i++)
    {
        for (size_t j = 0; j < block_size; j++)
        {
            printf("block: %d, t: %d, val: %f\n", i, j, h_vec[i * block_size + j]);
        }
        printf("\n");
    }
    free(h_vec);
}

void Utils::print_vec_mpi(
    double *vec, int len, int block_size, int rank, int world_size, std::string name)
{
    if (rank == 0)
    {
        printf("%s:\n", name.c_str());
    }
    for (int r = 0; r < world_size; r++)
    {
        if (rank == r)
        {
            printf("Rank: %d\n", r);
            print_vec(vec, len, block_size);
        }
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
}

void Utils::print_vec_complex(Complex *vec, int len, int block_size, std::string name)
{

    Complex *h_vec;
    h_vec = (Complex *)malloc((size_t)len * block_size * sizeof(Complex));
    gpuErrchk(
        cudaMemcpy(h_vec, vec, (size_t)len * block_size * sizeof(Complex), cudaMemcpyDeviceToHost));

    printf("%s:\n", name.c_str());

    for (size_t i = 0; i < len; i++)
    {
        for (size_t j = 0; j < block_size; j++)
        {
            printf("block: %d, t: %d, val: %f + %f i\n", i, j, h_vec[i * block_size + j].x,
                   h_vec[i * block_size + j].y);
        }
        printf("\n");
    }
    free(h_vec);
}

void Utils::make_table(std::vector<std::string> col_names, std::vector<long double> mean,
                       std::vector<long double> min, std::vector<long double> max)
{

    int size = col_names.size();

    if (mean.size() != size - 1 || min.size() != size - 1 || max.size() != size - 1)
    {
        std::cerr << "Error: make_table: input vectors must have the same size" << std::endl;
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        return;
    }

    tabulate::Table table;
    // table.add_row({title});
    table.format().font_align(tabulate::FontAlign::center);
    table.format().font_style({tabulate::FontStyle::bold});

    tabulate::Table::Row_t col_names_row(col_names.begin(), col_names.end());
    table.add_row(col_names_row);
    table[0][0]
        .format()
        .font_color(tabulate::Color::yellow)
        .font_style({tabulate::FontStyle::bold, tabulate::FontStyle::underline});

    // convert long double vectors to string vectors and add the row titles first
    std::vector<std::string> mean_str, min_str, max_str;
    mean_str.push_back("Mean Time (s)");
    min_str.push_back("Min Time (s)");
    max_str.push_back("Max Time (s)");

    for (int i = 0; i < size - 1; i++)
    {
        mean_str.push_back(std::to_string(mean[i]));
        min_str.push_back(std::to_string(min[i]));
        max_str.push_back(std::to_string(max[i]));
    }

    tabulate::Table::Row_t min_row(min_str.begin(), min_str.end());
    tabulate::Table::Row_t mean_row(mean_str.begin(), mean_str.end());
    tabulate::Table::Row_t max_row(max_str.begin(), max_str.end());

    table.add_row(min_row);
    table.add_row(mean_row);
    table.add_row(max_row);

    std::cout << table << std::endl;
}

void Utils::print_raw(long double *mean_times, long double *min_times, long double *max_times,
                      long double *mean_times_f, long double *min_times_f, long double *max_times_f,
                      long double *mean_times_fs, long double *min_times_fs, long double *max_times_fs, int times_len)
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

void Utils::print_times(int reps, bool table)
{
#if TIME_MPI

    int world_rank, world_size;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    int times_len = t_list_f.size();
    long double mpi_times_f[times_len], mpi_times_fs[times_len], mpi_times[3];
    long double *mean_times, *min_times, *max_times, *mean_times_f, *min_times_f, *max_times_f,
        *mean_times_fs, *min_times_fs, *max_times_fs;

    if (world_rank == 0)
    {
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

    for (int i = 0; i < times_len; i++)
    {
        mpi_times_f[i] = t_list_f[i].seconds / reps;
        mpi_times_fs[i] = t_list_fs[i].seconds / reps;
    }

    mpi_times[0] = t_list[ProfilerTimesFull::COMM_INIT].seconds;
    mpi_times[1] = t_list[ProfilerTimesFull::SETUP].seconds;
    mpi_times[2] = t_list[ProfilerTimesFull::FULL].seconds;

    MPICHECK(MPI_Reduce(
        (void *)&mpi_times, (void *)mean_times, 3, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(
        (void *)&mpi_times, (void *)min_times, 3, MPI_LONG_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce(
        (void *)&mpi_times, (void *)max_times, 3, MPI_LONG_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void *)&mpi_times_f, (void *)mean_times_f, times_len, MPI_LONG_DOUBLE,
                        MPI_SUM, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void *)&mpi_times_f, (void *)min_times_f, times_len, MPI_LONG_DOUBLE,
                        MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void *)&mpi_times_f, (void *)max_times_f, times_len, MPI_LONG_DOUBLE,
                        MPI_MAX, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void *)&mpi_times_fs, (void *)mean_times_fs, times_len, MPI_LONG_DOUBLE,
                        MPI_SUM, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void *)&mpi_times_fs, (void *)min_times_fs, times_len, MPI_LONG_DOUBLE,
                        MPI_MIN, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Reduce((void *)&mpi_times_fs, (void *)max_times_fs, times_len, MPI_LONG_DOUBLE,
                        MPI_MAX, 0, MPI_COMM_WORLD));

    if (world_rank == 0)
    {
        for (int i = 0; i < 3; i++)
        {
            mean_times[i] /= world_size;
        }
        for (int i = 0; i < times_len; i++)
        {
            mean_times_f[i] /= world_size;
            mean_times_fs[i] /= world_size;
        }
        if (table)
        {
            std::cout << "Aggregate Times: " << std::endl;
            std::cout << std::endl;

            make_table({"Summary", "Initialize", "Setup", "Matvecs"},
                       {mean_times[0], mean_times[1], mean_times[2]},
                       {min_times[0], min_times[1], min_times[2]},
                       {max_times[0], max_times[1], max_times[2]});
            std::vector<long double> mean_times_v(mean_times_f, mean_times_f + times_len);
            std::vector<long double> min_times_v(min_times_f, min_times_f + times_len);
            std::vector<long double> max_times_v(max_times_f, max_times_f + times_len);

            std::cout << std::endl;

            std::cout << "Average Times per Matvec: " << std::endl;

            std::cout << std::endl;

            make_table({"F Matvec", "Broadcast", "Pad", "FFT", "SOTI-to-TOSI", "SBGEMV",
                        "TOSI-to-SOTI", "IFFT", "Unpad", "Reduce", "Total"},
                       mean_times_v, min_times_v, max_times_v);

            std::cout << std::endl;

            mean_times_v = std::vector<long double>(mean_times_fs, mean_times_fs + times_len);
            min_times_v = std::vector<long double>(min_times_fs, min_times_fs + times_len);
            max_times_v = std::vector<long double>(max_times_fs, max_times_fs + times_len);

            make_table({"F* Matvec", "Broadcast", "Pad", "FFT", "SOTI-to-TOSI", "SBGEMV",
                        "TOSI-to-SOTI", "IFFT", "Unpad", "Reduce", "Total"},
                       mean_times_v, min_times_v, max_times_v);
        }
        else
        {
            print_raw(mean_times, min_times, max_times, mean_times_f, min_times_f, max_times_f,
                      mean_times_fs, min_times_fs, max_times_fs, times_len);
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

void Utils::swap_axes(
    Complex *d_in, Complex *d_out, int num_cols, int num_rows, int block_size, cudaStream_t s)
{
#if CUTENSOR_AVAILABLE
    // use cuTensor to swap axes d_in[t,m,d] -> d_out[d,m,t] (column-major)

    cutensorDataType_t typeA = CUTENSOR_C_64F;
    cutensorDataType_t typeB = CUTENSOR_C_64F;

    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_64F;

    double alpha = 1.0;

    std::vector<int> modeA = {'t', 'm', 'd'};
    std::vector<int> modeB = {'d', 'm', 't'};

    int nmode = 3;

    std::vector<int64_t> extentA = {block_size, num_cols, num_rows};
    std::vector<int64_t> extentB = {num_rows, num_cols, block_size};

    cutensorHandle_t handle;
    cutensorSafeCall(cutensorCreate(&handle));

    uint32_t const kAlignment = 128;
    assert(uintptr_t(d_in) % kAlignment == 0);
    assert(uintptr_t(d_out) % kAlignment == 0);

    cutensorTensorDescriptor_t descA;

    cutensorSafeCall(cutensorCreateTensorDescriptor(
        handle, &descA, nmode, extentA.data(), nullptr, typeA, kAlignment));

    cutensorTensorDescriptor_t descB;

    cutensorSafeCall(cutensorCreateTensorDescriptor(
        handle, &descB, nmode, extentB.data(), nullptr, typeB, kAlignment));

    cutensorOperationDescriptor_t desc;
    cutensorSafeCall(cutensorCreatePermutation(handle, &desc, descA, modeA.data(),
                                               CUTENSOR_OP_IDENTITY, descB, modeB.data(), descCompute));

    cutensorDataType_t scalarType;
    cutensorSafeCall(cutensorOperationDescriptorGetAttribute(handle, desc,
                                                             CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, (void *)&scalarType, sizeof(scalarType)));

    assert(scalarType == CUTENSOR_C_64F);

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;

    cutensorSafeCall(cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE));

    cutensorPlan_t plan;
    cutensorSafeCall(cutensorCreatePlan(handle, &plan, desc, planPref, 0 /* worksize */));

    cutensorSafeCall(cutensorPermute(handle, plan, &alpha, d_in, d_out, s));

    cutensorSafeCall(cutensorDestroy(handle));
    cutensorSafeCall(cutensorDestroyPlan(plan));
    cutensorSafeCall(cutensorDestroyOperationDescriptor(desc));
    cutensorSafeCall(cutensorDestroyPlanPreference(planPref));
    cutensorSafeCall(cutensorDestroyTensorDescriptor(descA));
    cutensorSafeCall(cutensorDestroyTensorDescriptor(descB));
#else
    // use cutranspose to swap axes d_in[t,m,d] -> d_out[d,m,t] (column-major)
    UtilKernels::swap_axes_cutranspose(d_in, d_out, num_cols, num_rows, block_size, s);
#endif

}

void Utils::check_collective_io(const HighFive::DataTransferProps &xfer_props)
{
    auto mnccp = HighFive::MpioNoCollectiveCause(xfer_props);
    if (mnccp.getLocalCause() || mnccp.getGlobalCause())
    {
        std::cout
            << "The operation was successful, but couldn't use collective MPI-IO. local cause: "
            << mnccp.getLocalCause() << " global cause:" << mnccp.getGlobalCause() << std::endl;
    }
}

size_t Utils::get_start_index(size_t glob_num_blocks, int color, int comm_size)
{
    return (color < glob_num_blocks % comm_size)
               ? (glob_num_blocks / comm_size + 1) * color
               : (glob_num_blocks / comm_size) * color + glob_num_blocks % comm_size;
}

int Utils::global_to_local_size(int global_size, int color, int comm_size)
{
    if (color >= comm_size)
    {
        fprintf(stderr, "Invalid color for communicator. Got color = %d, comm_size = %d\n", color,
                comm_size);
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }
    if (global_size < comm_size)
    {
        fprintf(stderr,
                "Make sure global_size >= comm_size. Got global_size = %d, comm_size = %d\n",
                global_size, comm_size);
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }
    return (color < global_size % comm_size) ? global_size / comm_size + 1
                                             : global_size / comm_size;
}

int Utils::local_to_global_size(int local_size, int comm_size) { return local_size * comm_size; }

std::string Utils::zero_pad(size_t num, size_t width)
{
    std::string num_str = std::to_string(num);
    return std::string(width - std::min(width, num_str.length()), '0') + num_str;
}
