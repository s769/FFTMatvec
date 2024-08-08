
#include "shared.hpp"
#include "utils.cuh"
#include "cmdparser.hpp"
#include "matvec.hpp"
#include "Comm.hpp"
#include "Vector.hpp"
#include "Matrix.hpp"









// command line arguments parser

void configureParser(cli::Parser &parser)
{
  parser.set_optional<int>("pr", "proc_rows", 1, "Number of processor rows");
  parser.set_optional<int>("pc", "proc_cols", 1, "Number of processor columns");
  parser.set_optional<bool>("g", "glob_inds", false, "Use global indices");
  parser.set_optional<int>("Nm", "glob_cols", 10, "Number of global columns");
  parser.set_optional<int>("Nd", "glob_rows", 5, "Number of global rows");
  parser.set_optional<int>("Nt", "unpad_size", 7, "Number of time points");
  parser.set_optional<int>("nm", "num_cols", 3, "Number of local columns");
  parser.set_optional<int>("nd", "num_rows", 2, "Number of local rows");
  parser.set_optional<bool>("p", "print", false, "Print vectors");
}

/********/
/* MAIN */
/********/
int main(int argc, char **argv)
{

  int world_rank, world_size, provided;

  MPICHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

  if (provided < MPI_THREAD_FUNNELED)
  {
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


  if (world_size != proc_rows * proc_cols)
  {
    if (world_rank == 0)
      fprintf(stderr, "Proc Rows x Proc Cols must equal the total number of MPI ranks\n");
    MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    return 1;
  }

  int row_color = world_rank % proc_rows; //(conjugate) ? world_rank % proc_cols : world_rank / proc_cols;
  int col_color = world_rank / proc_rows; //(conjugate) ? world_rank / proc_cols : world_rank % proc_cols;


  auto glob_inds = parser.get<bool>("g");

  auto unpad_size = parser.get<int>("Nt");
  auto glob_num_cols = parser.get<int>("Nm");
  auto glob_num_rows = parser.get<int>("Nd");
  auto num_cols = parser.get<int>("nm");
  auto num_rows = parser.get<int>("nd");

  if (!glob_inds)
  {
    glob_num_cols = num_cols * proc_cols;
    glob_num_rows = num_rows * proc_rows;
  }


  if (world_rank == 0)
  {
    printf("Proc Rows: %d, Proc Cols: %d\n", proc_rows, proc_cols);
    printf("Global Rows: %d, Global Cols: %d\n", glob_num_rows, glob_num_cols);
    printf("Unpad Size: %d\n", unpad_size);
  }

  // double *h_mat;

  if (!glob_inds)
  {
    num_cols = (col_color < glob_num_cols % proc_cols) ? glob_num_cols / proc_cols + 1 : glob_num_cols / proc_cols;
    num_rows = (row_color < glob_num_rows % proc_rows) ? glob_num_rows / proc_rows + 1 : glob_num_rows / proc_rows;
  }
  if (proc_rows > glob_num_rows)
  {

    if (world_rank == 0)
      fprintf(stderr,"make sure proc_rows <= glob_num_rows \n");
    MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    return 1;
  }

  if (proc_cols > glob_num_cols)
  {
    if (world_rank == 0)
      fprintf(stderr,"make sure proc_cols <= glob_num_cols \n");
    MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    return 1;
  }

  unsigned int size = 2 * unpad_size;


  Comm comm(MPI_COMM_WORLD, proc_rows, proc_cols);

  if (world_rank == 0) printf("Created Comm\n");

  Matrix F(comm, num_cols, num_rows, size, false, false);

  if (world_rank == 0) printf("Created Matrices\n");

  F.init_mat_ones();

  if (world_rank == 0) printf("Initialized Matrices\n");

  Vector in_F(comm, num_cols, size, "row"), in_FS(comm, num_rows, size, "col");
  Vector out_F(in_FS), out_FS(in_F);

  if (world_rank == 0) printf("Created Vectors\n");

  in_F.init_vec_ones();
  in_FS.init_vec_ones();
  out_F.init_vec();
  out_FS.init_vec();

  bool print = parser.get<bool>("p");

  if (print)
  {
    in_F.print("in_F");
    in_FS.print("in_FS");
  }


  if (world_rank == 0) printf("Initialized Vectors\n");



  F.matvec(in_F, out_F);
  F.transpose_matvec(in_FS, out_FS);

  if (print)
  {
    out_F.print("out_F");
    out_FS.print("out_FS");
  }

  F.matvec(in_F, out_FS, true);
  F.transpose_matvec(in_FS, out_F, true);

  if (print)
  {
    out_F.print("out_F");
    out_FS.print("out_FS");
  }



  }
  MPICHECK(MPI_Finalize());
  
  return 0;
}