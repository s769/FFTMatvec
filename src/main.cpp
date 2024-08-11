
#include "Comm.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include "cmdparser.hpp"
#include "matvec.hpp"
#include "shared.hpp"
#include "utils.hpp"
#include "tester.hpp"

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
    parser.set_optional<int>("pr", "proc_rows", 1, "Number of processor rows");
    parser.set_optional<int>("pc", "proc_cols", 1, "Number of processor columns");
    parser.set_optional<bool>("g", "glob_inds", false, "Use global indices");
    parser.set_optional<int>("Nm", "glob_cols", 10, "Number of global columns");
    parser.set_optional<int>("Nd", "glob_rows", 5, "Number of global rows");
    parser.set_optional<int>("Nt", "unpad_size", 7, "Number of time points");
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

    int world_rank, world_size, provided;

    MPICHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0)
            fprintf(stderr, "The provided MPI level is not sufficient\n");
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
        return 1;
    }

    {

        cli::Parser parser(argc, argv);
        configureParser(parser);
        parser.run_and_exit_if_error();

        MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
        MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

        auto proc_rows = parser.get<int>("pr");
        auto proc_cols = parser.get<int>("pc");

        if (world_size != proc_rows * proc_cols) {
            if (world_rank == 0)
                fprintf(stderr, "Proc Rows x Proc Cols must equal the total number of MPI ranks\n");
            MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
            return 1;
        }

        int row_color = world_rank
            % proc_rows; //(conjugate) ? world_rank % proc_cols : world_rank / proc_cols;
        int col_color = world_rank
            / proc_rows; //(conjugate) ? world_rank / proc_cols : world_rank % proc_cols;

        auto glob_inds = parser.get<bool>("g");

        auto unpad_size = parser.get<int>("Nt");
        auto glob_num_cols = parser.get<int>("Nm");
        auto glob_num_rows = parser.get<int>("Nd");
        auto num_cols = parser.get<int>("nm");
        auto num_rows = parser.get<int>("nd");

        if (!glob_inds) {
            glob_num_cols = num_cols * proc_cols;
            glob_num_rows = num_rows * proc_rows;
        }

        if (world_rank == 0) {
            printf("Proc Rows: %d, Proc Cols: %d\n", proc_rows, proc_cols);
            printf("Global Rows: %d, Global Cols: %d\n", glob_num_rows, glob_num_cols);
            printf("Unpad Size: %d\n", unpad_size);
        }

        if (!glob_inds) {
            num_cols = (col_color < glob_num_cols % proc_cols) ? glob_num_cols / proc_cols + 1
                                                               : glob_num_cols / proc_cols;
            num_rows = (row_color < glob_num_rows % proc_rows) ? glob_num_rows / proc_rows + 1
                                                               : glob_num_rows / proc_rows;
        }
        if (proc_rows > glob_num_rows) {

            if (world_rank == 0)
                fprintf(stderr, "make sure proc_rows <= glob_num_rows \n");
            MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
            return 1;
        }

        if (proc_cols > glob_num_cols) {
            if (world_rank == 0)
                fprintf(stderr, "make sure proc_cols <= glob_num_cols \n");
            MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
            return 1;
        }

        unsigned int size = 2 * unpad_size;

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_list[ProfilerTimesFull::COMM_INIT].start();
#endif
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
        Matrix F(comm, num_cols, num_rows, size, false, false);

        if (world_rank == 0)
            printf("Created Matrices\n");

        F.init_mat_ones();

        if (world_rank == 0)
            printf("Initialized Matrices\n");

        Vector in_F(comm, num_cols, size, "row"), in_FS(comm, num_rows, size, "col");
        Vector out_F(in_FS), out_FS(in_F);

        if (world_rank == 0)
            printf("Created Vectors\n");

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

        for (int i = 0; i < reps + WARMUP; i++) {
            F.matvec(in_F, out_F);
            F.transpose_matvec(in_FS, out_FS);
        }

#if TIME_MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_list[ProfilerTimesFull::FULL].stop();
#endif

        auto test = parser.get<bool>("t");

        if (test) {
            Tester::checkOnesMatvec(comm, F, out_F, false, false);
            Tester::checkOnesMatvec(comm, F, out_FS, true, false);
        }


        if (world_rank == 0)
            printf("Finished Matvecs\n");
        if (print) {
            out_F.print("out_F");
            out_FS.print("out_FS");
        }

#if !TIME_MPI

        F.matvec(in_F, out_FS, true);
        F.transpose_matvec(in_FS, out_F, true);

        if (print) {
            out_F.print("out_F");
            out_FS.print("out_FS");
        }

#endif

        auto print_raw = parser.get<bool>("raw");
        if (world_rank == 0)
            printf("Timing Results Showing Mean, Min, and Max Times Over %d Processor(s) (%d "
                   "Matvecs):\n\n",
                world_size, reps);

        Utils::printTimes(reps, !print_raw);
    }
    MPICHECK(MPI_Finalize());

    return 0;
}