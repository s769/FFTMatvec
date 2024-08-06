
#include "shared.cuh"
#include "reduction.cuh"
#include "multiply.cuh"
#include "utils.cuh"
#include "matfuncs.cuh"
#include "cmdparser.hpp"
#include "matvec.cuh"









// command line arguments parser

void configureParser(cli::Parser &parser)
{
  parser.set_required<int>("pr", "proc_rows", "Number of processor rows");
  parser.set_required<int>("pc", "proc_cols", "Number of processor columns");
  parser.set_optional<bool>("g", "glob_inds", false, "Use global indices");
  parser.set_optional<int>("Nm", "glob_cols", 10, "Number of global columns");
  parser.set_optional<int>("Nd", "glob_rows", 5, "Number of global rows");
  parser.set_optional<int>("Nt", "unpad_size", 7, "Number of time points");
  parser.set_optional<int>("nm", "num_cols", 3, "Number of local columns");
  parser.set_optional<int>("nr", "num_rows", 2, "Number of local rows");
}

/********/
/* MAIN */
/********/
int main(int argc, char **argv)
{

  int world_rank, nRanks;

  MPICHECK(MPI_Init(&argc, &argv));

  cli::Parser parser(argc, argv);
  configureParser(parser);
  parser.run_and_exit_if_error();

  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  auto proc_rows = parser.get<int>("proc_rows");
  auto proc_cols = parser.get<int>("proc_cols");

  if (nRanks != proc_rows * proc_cols)
  {
    if (world_rank == 0)
      printf("Proc Rows x Proc Cols must equal the total number of MPI ranks\n");
    MPICHECK(MPI_Finalize());
    return 1;
  }

  int row_color = world_rank % proc_rows; //(conjugate) ? world_rank % proc_cols : world_rank / proc_cols;
  int col_color = world_rank / proc_rows; //(conjugate) ? world_rank / proc_cols : world_rank % proc_cols;

  bool newmv = true;

  auto glob_inds = parser.get<bool>("glob_inds");

  int glob_num_cols, glob_num_rows, unpad_size, num_cols, num_rows;

  if (glob_inds)
  {
    glob_num_cols = parser.get<int>("glob_cols");
    glob_num_rows = parser.get<int>("glob_rows");
    unpad_size = parser.get<int>("unpad_size");
  }
  else
  {
    num_cols = parser.get<int>("num_cols");
    num_rows = parser.get<int>("num_rows");
    unpad_size = parser.get<int>("unpad_size");
    glob_num_cols = num_cols * proc_cols;
    glob_num_rows = num_rows * proc_rows;
  }

  double *h_mat;

  if (!glob_inds)
  {
    num_cols = (col_color < glob_num_cols % proc_cols) ? glob_num_cols / proc_cols + 1 : glob_num_cols / proc_cols;
    num_rows = (row_color < glob_num_rows % proc_rows) ? glob_num_rows / proc_rows + 1 : glob_num_rows / proc_rows;
  }
  if (proc_rows > glob_num_rows)
  {

    if (world_rank == 0)
      printf("make sure proc_rows <= glob_num_rows \n");
    MPICHECK(MPI_Finalize());
    return 1;
  }

  if (proc_cols > glob_num_cols)
  {
    if (world_rank == 0)
      printf("make sure proc_cols <= glob_num_cols \n");
    MPICHECK(MPI_Finalize());
    return 1;
  }

  unsigned int size = 2 * unpad_size;

  MPI_Comm group_comm, group_comm2;
  Comm_t nccl_comm, nccl_comm2;
  cudaStream_t s;
  int device;
  init_comms(MPI_COMM_WORLD, &group_comm, &group_comm2, &nccl_comm, &nccl_comm2, &s, &device, proc_rows, proc_cols, false);

  Complex *d_mat_freq;

  init_hmat(num_rows, num_cols, size, &h_mat);

  setup_new(&d_mat_freq, h_mat, size, num_cols, num_rows);
  free(h_mat);



  double noise_scale = 1; // e-6;

  matvec_args_t args_f;
  args_f.d_mat_freq = d_mat_freq;
  args_f.size = size;
  args_f.num_cols = num_cols;
  args_f.num_rows = num_rows;
  args_f.device = device;
  args_f.noise_scale = noise_scale;
  args_f.row_comm = nccl_comm;
  args_f.col_comm = nccl_comm2;
  args_f.row_color = row_color;
  args_f.col_color = col_color;

  matvec_args_t args_fs;
  args_fs.d_mat_freq = d_mat_freq;
  args_fs.size = size;
  args_fs.num_cols = num_cols;
  args_fs.num_rows = num_rows;
  args_fs.device = device;
  args_fs.noise_scale = noise_scale;
  args_fs.row_comm = nccl_comm;
  args_fs.col_comm = nccl_comm2;
  args_fs.row_color = row_color;
  args_fs.col_color = col_color;




  createMat(&args_f, row_color, col_color, size, num_rows, num_cols, proc_rows, proc_cols, false, false, s, (bool)newmv);
  createMat(&args_fs, col_color, row_color, size, num_rows, num_cols, proc_cols, proc_rows, true, false, s, (bool)newmv);

  double *d_in_f, *d_in_fs, *d_out_f, *d_out_fs;

  (row_color == 0) ? init_vector(num_cols, unpad_size, &d_in_f, true) : init_vector(num_cols, unpad_size, &d_in_f, false);
  (col_color == 0) ? init_vector(num_rows, unpad_size, &d_in_fs, true) : init_vector(num_rows, unpad_size, &d_in_fs, false);

  init_vector(num_rows, unpad_size, &d_out_f, false);
  init_vector(num_cols, unpad_size, &d_out_fs, false);


  MatVec(&args_f, d_in_f, d_out_f, false, false);
  MatVec(&args_fs, d_in_fs, d_out_fs, true, false);



  destroyMat(&args_f, row_color, col_color, false, false, (bool)newmv);
  destroyMat(&args_fs, col_color, row_color, true, false, (bool)newmv);

  gpuErrchk(cudaFree(d_in_f));
  gpuErrchk(cudaFree(d_in_fs));
  gpuErrchk(cudaFree(d_out_f));
  gpuErrchk(cudaFree(d_out_fs));





  NCCLCHECK(ncclCommDestroy(nccl_comm2));
  MPICHECK(MPI_Comm_free(&group_comm2));

  gpuErrchk(cudaFree(d_mat_freq));


  NCCLCHECK(ncclCommDestroy(nccl_comm));
  MPICHECK(MPI_Comm_free(&group_comm));

  MPICHECK(MPI_Finalize());

  return 0;
}