
/**
 * @file main.cpp
 * @brief This file contains the main function for the matvec-test program.
 *
 * The program performs matrix-vector multiplication using MPI parallelization.
 * It takes command line arguments to configure the matrix and vector dimensions,
 * the number of processor rows and columns, and other options.
 *
 * The main function initializes MPI, parses command line arguments, creates a communication object,
 * initializes matrices and vectors, performs matrix-vector multiplication, and prints the results.
 *
 * @param argc The number of command line arguments.
 * @param argv An array of command line argument strings.
 * @return 0 on success, 1 on error.
 */
#include "Comm.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include "cmdparser.hpp"
#include "matvec.hpp"
#include "shared.hpp"
#include "tester.hpp"
#include "utils.hpp"

// enable/disable timing (see shared.hpp)
#if TIME_MPI
#define WARMUP 10
bool warmup;
#endif

/**
 * @brief Configures the parser.
 * @param parser The parser.
 */

void configureParser(cli::Parser& parser)
{
    parser.set_optional<int>(
        "pr", "proc_rows", 1, "Number of processor rows (proc_rows x proc_cols = num_mpi_ranks)");
    parser.set_optional<int>("pc", "proc_cols", 1,
        "Number of processor columns (proc_rows x proc_cols = num_mpi_ranks)");
    parser.set_optional<bool>("g", "glob_inds", false, "Use global indices");
    parser.set_optional<int>("Nm", "glob_cols", 10, "Number of global columns");
    parser.set_optional<int>("Nd", "glob_rows", 5, "Number of global rows");
    parser.set_optional<int>("Nt", "block_size", 7, "Number of time points");
    parser.set_optional<int>("nm", "num_cols", 3, "Number of local columns");
    parser.set_optional<int>("nd", "num_rows", 2, "Number of local rows");
    parser.set_optional<bool>("v", "verbose", false, "Print vectors");
    parser.set_optional<int>("N", "reps", 100, "Number of repetitions (for timing purposes)");
    parser.set_optional<bool>("raw", "print_raw", false, "Print raw times (instead of table)");
    parser.set_optional<bool>("t", "test", false, "Run tests");
}

/********/
/* MAIN */
/********/
int main(int argc, char** argv)
{
    int world_rank = 0, world_size, provided;
    // Initialize the MPI environment (OpenMP is used so we need to use MPI_Init_thread)
    MPICHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0)
            fprintf(stderr, "The provided MPI level is not sufficient\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        return 1;
    }

    {
        // Parse command line arguments
        cli::Parser parser(argc, argv);
        configureParser(parser);
        parser.run_and_exit_if_error();

        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

        auto proc_rows = parser.get<int>("pr");
        auto proc_cols = parser.get<int>("pc");
        // Check that the number of processor rows and columns is valid
        if (world_size != proc_rows * proc_cols) {
            if (proc_rows == 1 && proc_cols == 1) {
                proc_rows = 1;
                proc_cols = world_size;
                fprintf(stderr,
                    "Warning: Proc Rows x Proc Cols must equal the total number of MPI "
                    "ranks. Got %d x %d = %d, expected %d. Using %d x %d = %d instead\n",
                    proc_rows, proc_cols, proc_rows * proc_cols, world_size, proc_rows, proc_cols,
                    proc_rows * proc_cols);
            } else if (world_rank == 0) {
                fprintf(stderr,
                    "Proc Rows x Proc Cols must equal the total number of MPI ranks. Got %d x %d = "
                    "%d, expected %d\n",
                    proc_rows, proc_cols, proc_rows * proc_cols, world_size);
                MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
                return 1;
            }
        }

        int row_color = world_rank % proc_rows;
        int col_color = world_rank / proc_rows;

        auto glob_inds = parser.get<bool>("g");

        auto block_size = parser.get<int>("Nt");
        auto glob_num_cols = parser.get<int>("Nm");
        auto glob_num_rows = parser.get<int>("Nd");
        auto num_cols = parser.get<int>("nm");
        auto num_rows = parser.get<int>("nd");
        // If not using global indices, calculate the number of global columns and rows
        if (!glob_inds) {
            glob_num_cols = num_cols * proc_cols;
            glob_num_rows = num_rows * proc_rows;
        }

        // If using global indices, calculate the number of local columns and rows
        if (glob_inds) {
            num_cols = (col_color < glob_num_cols % proc_cols) ? glob_num_cols / proc_cols + 1
                                                               : glob_num_cols / proc_cols;
            num_rows = (row_color < glob_num_rows % proc_rows) ? glob_num_rows / proc_rows + 1
                                                               : glob_num_rows / proc_rows;
        }
        // Check that the number of local columns and rows is valid when using global indices
        if (glob_inds && proc_rows > glob_num_rows) {

            if (world_rank == 0)
                fprintf(stderr, "make sure proc_rows <= glob_num_rows \n");
            MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
            return 1;
        }

        if (glob_inds && proc_cols > glob_num_cols) {
            if (world_rank == 0)
                fprintf(stderr, "make sure proc_cols <= glob_num_cols \n");
            MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
            return 1;
        }


        if (world_rank == 0) {
            printf("Proc Rows: %d, Proc Cols: %d\n", proc_rows, proc_cols);
            printf("Global Rows: %d, Global Cols: %d\n", glob_num_rows, glob_num_cols);
            printf("Block Size: %d\n", block_size);
        }

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_list[ProfilerTimesFull::COMM_INIT].start();
#endif
        // Create a communicator object
        Comm comm(MPI_COMM_WORLD, proc_rows, proc_cols);
#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_list[ProfilerTimesFull::COMM_INIT].stop();
#endif

        if (world_rank == 0)
            printf("Created Comm\n");

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_list[ProfilerTimesFull::SETUP].start();
#endif
        // Create matrices and vectors
        Matrix F(comm, num_cols, num_rows, block_size);

        if (world_rank == 0)
            printf("Created Matrices\n");

        F.init_mat_ones();

        if (world_rank == 0)
            printf("Initialized Matrices\n");

        Vector in_F(comm, num_cols, block_size, "col"), in_FS(comm, num_rows, block_size, "row");
        Vector out_F(in_FS), out_FS(in_F);

        if (world_rank == 0)
            printf("Created Vectors\n");
        // Initialize vectors with ones for testing
        in_F.init_vec_ones();
        in_FS.init_vec_ones();
        out_F.init_vec();
        out_FS.init_vec();

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_list[ProfilerTimesFull::SETUP].stop();
#endif

        bool print = parser.get<bool>("v");

        if (print) {
            in_F.print("in_F");
            in_FS.print("in_FS");
        }


        double in_F_norm_2 = in_F.norm(2);
        double in_FS_norm_2 = in_FS.norm(2);
        double in_F_norm_1 = in_F.norm(1);
        double in_FS_norm_1 = in_FS.norm(1);
        double in_F_norm_inf = in_F.norm(-1);
        double in_FS_norm_inf = in_FS.norm(-1);
        if (world_rank == 0){
            std::cout << "||in_F||_2 = " << in_F_norm_2 << std::endl;
            std::cout << "||in_FS||_2 = " << in_FS_norm_2 << std::endl;
            std::cout << "||in_F||_1 = " << in_F_norm_1 << std::endl;
            std::cout << "||in_FS||_1 = " << in_FS_norm_1 << std::endl;
            std::cout << "||in_F||_inf = " << in_F_norm_inf << std::endl;
            std::cout << "||in_FS||_inf = " << in_FS_norm_inf << std::endl;
        }

        if (world_rank == 0)
            printf("Initialized Vectors\n");

        auto reps = parser.get<int>("N");

#if !TIME_MPI
        reps = 1;
#endif

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_list[ProfilerTimesFull::FULL].start();
#endif
        // Perform matrix-vector multiplication
        for (int i = 0; i < reps + WARMUP; i++) {
            F.matvec(in_F, out_F);
            F.transpose_matvec(in_FS, out_FS);
        }

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_list[ProfilerTimesFull::FULL].stop();
#endif

        auto test = parser.get<bool>("t");
        // Run tests
        if (test) {
            Tester::check_ones_matvec(comm, F, out_F, false, false);
            Tester::check_ones_matvec(comm, F, out_FS, true, false);
        }

        if (world_rank == 0)
            printf("Finished Matvecs\n");
        if (print) {
            out_F.print("out_F");
            out_FS.print("out_FS");
        }

        double out_F_norm_2 = out_F.norm(2);
        double out_FS_norm_2 = out_FS.norm(2);
        double out_F_norm_1 = out_F.norm(1);
        double out_FS_norm_1 = out_FS.norm(1);
        double out_F_norm_inf = out_F.norm(-1);
        double out_FS_norm_inf = out_FS.norm(-1);
        double dot_inF_outFS = in_F.dot(out_FS);
        double dot_inFS_outF = in_FS.dot(out_F);

        if (world_rank == 0){
            std::cout << "||out_F||_2 = " << out_F_norm_2 << std::endl;
            std::cout << "||out_FS||_2 = " << out_FS_norm_2 << std::endl;
            std::cout << "||out_F||_1 = " << out_F_norm_1 << std::endl;
            std::cout << "||out_FS||_1 = " << out_FS_norm_1 << std::endl;
            std::cout << "||out_F||_inf = " << out_F_norm_inf << std::endl;
            std::cout << "||out_FS||_inf = " << out_FS_norm_inf << std::endl;
            std::cout << "<in_F, out_FS> = " << dot_inF_outFS << std::endl;
            std::cout << "<in_FS, out_F> = " << dot_inFS_outF << std::endl;
        }

#if !TIME_MPI

            F.matvec(in_F, out_FS, true);
        F.transpose_matvec(in_FS, out_F, true);

        if (print) {
            out_F.print("out_F");
            out_FS.print("out_FS");
        }

#endif
        // Print timing results
        auto print_raw = parser.get<bool>("raw");
        if (world_rank == 0)
            printf("Timing Results Showing Mean, Min, and Max Times Over %d Processor(s) (%d "
                   "Matvecs):\n\n",
                world_size, reps);

        Utils::print_times(reps, !print_raw);
    }
    MPICHECK(MPI_Finalize());

    return 0;
}
