/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define LOCALSIZE       64
#define LOCALSIZE2      1
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold OpenCL objects */
typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  propagate;

  cl_mem speeds0;
  cl_mem speeds1;
  cl_mem speeds2;
  cl_mem speeds3;
  cl_mem speeds4;
  cl_mem speeds5;
  cl_mem speeds6;
  cl_mem speeds7;
  cl_mem speeds8;
  cl_mem tmp_speeds0;
  cl_mem tmp_speeds1;
  cl_mem tmp_speeds2;
  cl_mem tmp_speeds3;
  cl_mem tmp_speeds4;
  cl_mem tmp_speeds5;
  cl_mem tmp_speeds6;
  cl_mem tmp_speeds7;
  cl_mem tmp_speeds8;
  cl_mem partial_sum;
  cl_mem partial_sum2;

  cl_mem obstacles;
} t_ocl;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

typedef struct
{
  float* restrict s0 ;
  float* restrict s1 ;
  float* restrict s2 ;
  float* restrict s3 ;
  float* restrict s4 ;
  float* restrict s5 ;
  float* restrict s6 ;
  float* restrict s7 ;
  float* restrict s8 ;
} t_speeds;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl* ocl);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_ocl    ocl;                 /* struct to hold OpenCL objects */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  cl_int err;
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

/* initialise our data structures and load values from file */
initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl);

t_speeds speeds;
speeds.s0 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
speeds.s1 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
speeds.s2 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
speeds.s3 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
speeds.s4 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
speeds.s5 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
speeds.s6 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
speeds.s7 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
speeds.s8 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
t_speeds tmp_speeds;
tmp_speeds.s0 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
tmp_speeds.s1 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
tmp_speeds.s2 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
tmp_speeds.s3 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
tmp_speeds.s4 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
tmp_speeds.s5 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
tmp_speeds.s6 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
tmp_speeds.s7 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);
tmp_speeds.s8 = _mm_malloc(sizeof(float) * (params.ny * params.nx),64);

/* loop over _all_ cells */
for (int jj = 0; jj < params.ny; jj++)
{
  for (int ii = 0; ii < params.nx; ii++)
  {
    speeds.s0[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[0];
    speeds.s1[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[1];
    speeds.s2[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[2];
    speeds.s3[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[3];
    speeds.s4[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[4];
    speeds.s5[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[5];
    speeds.s6[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[6];
    speeds.s7[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[7];
    speeds.s8[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[8];
    tmp_speeds.s0[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[0];
    tmp_speeds.s1[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[1];
    tmp_speeds.s2[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[2];
    tmp_speeds.s3[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[3];
    tmp_speeds.s4[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[4];
    tmp_speeds.s5[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[5];
    tmp_speeds.s6[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[6];
    tmp_speeds.s7[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[7];
    tmp_speeds.s8[ii + jj*params.nx] = cells[ii + jj*params.nx].speeds[8];
  }
}

/* iterate for maxIters timesteps */
gettimeofday(&timstr, NULL);
tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  // Write obstacles to OpenCL buffer
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.obstacles, CL_FALSE, 0,
    sizeof(cl_int) * params.nx * params.ny, obstacles, 0, NULL, NULL);
  checkError(err, "writing obstacles data", __LINE__);
  // Write cells to device
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.speeds0, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s0, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.speeds1, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s1, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.speeds2, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s2, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.speeds3, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s3, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.speeds4, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s4, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.speeds5, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s5, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.speeds6, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s6, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.speeds7, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s7, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.speeds8, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s8, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);

  float densityaccel = params.density*params.accel;

    // Set kernel arguments
  err = clSetKernelArg(ocl.propagate, 18, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting propagate arg 2", __LINE__);
  err = clSetKernelArg(ocl.propagate, 19, sizeof(cl_int), &params.nx);
  checkError(err, "setting propagate arg 3", __LINE__);
  err = clSetKernelArg(ocl.propagate, 20, sizeof(cl_int), &params.ny);
  checkError(err, "setting propagate arg 4", __LINE__);
  err = clSetKernelArg(ocl.propagate, 21, sizeof(cl_float), &params.omega);
  checkError(err, "setting propagate arg 5", __LINE__);
  err = clSetKernelArg(ocl.propagate, 22, sizeof(cl_float)*LOCALSIZE*LOCALSIZE2, NULL);
  checkError(err, "setting propagate arg 6", __LINE__);
  err = clSetKernelArg(ocl.propagate, 23, sizeof(cl_int)*LOCALSIZE*LOCALSIZE2, NULL);
  checkError(err, "setting propagate arg 7", __LINE__);
  err = clSetKernelArg(ocl.propagate, 24, sizeof(cl_mem), &ocl.partial_sum);
  checkError(err, "setting propagate arg 8", __LINE__);
  err = clSetKernelArg(ocl.propagate, 25, sizeof(cl_mem), &ocl.partial_sum2);
  checkError(err, "setting propagate arg 9", __LINE__);
  err = clSetKernelArg(ocl.propagate, 27, sizeof(cl_float), &densityaccel);
  checkError(err, "setting accelerate_flow arg 4", __LINE__);


  err = clFinish(ocl.queue);
  checkError(err, "waiting for propagate kernel", __LINE__);

  for (int tt = 0; tt < params.maxIters; tt++){
      cl_int err;

      // Set kernel arguments
      if(tt%2==0){
        err = clSetKernelArg(ocl.propagate, 0, sizeof(cl_mem), &ocl.speeds0);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 1, sizeof(cl_mem), &ocl.speeds1);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 2, sizeof(cl_mem), &ocl.speeds2);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 3, sizeof(cl_mem), &ocl.speeds3);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 4, sizeof(cl_mem), &ocl.speeds4);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 5, sizeof(cl_mem), &ocl.speeds5);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 6, sizeof(cl_mem), &ocl.speeds6);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 7, sizeof(cl_mem), &ocl.speeds7);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 8, sizeof(cl_mem), &ocl.speeds8);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 9, sizeof(cl_mem), &ocl.tmp_speeds0);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 10, sizeof(cl_mem), &ocl.tmp_speeds1);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 11, sizeof(cl_mem), &ocl.tmp_speeds2);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 12, sizeof(cl_mem), &ocl.tmp_speeds3);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 13, sizeof(cl_mem), &ocl.tmp_speeds4);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 14, sizeof(cl_mem), &ocl.tmp_speeds5);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 15, sizeof(cl_mem), &ocl.tmp_speeds6);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 16, sizeof(cl_mem), &ocl.tmp_speeds7);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 17, sizeof(cl_mem), &ocl.tmp_speeds8);
        checkError(err, "setting propagate arg 1", __LINE__);
      }else{
        err = clSetKernelArg(ocl.propagate, 0, sizeof(cl_mem), &ocl.tmp_speeds0);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 1, sizeof(cl_mem), &ocl.tmp_speeds1);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 2, sizeof(cl_mem), &ocl.tmp_speeds2);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 3, sizeof(cl_mem), &ocl.tmp_speeds3);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 4, sizeof(cl_mem), &ocl.tmp_speeds4);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 5, sizeof(cl_mem), &ocl.tmp_speeds5);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 6, sizeof(cl_mem), &ocl.tmp_speeds6);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 7, sizeof(cl_mem), &ocl.tmp_speeds7);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 8, sizeof(cl_mem), &ocl.tmp_speeds8);
        checkError(err, "setting propagate arg 0", __LINE__);
        err = clSetKernelArg(ocl.propagate, 9, sizeof(cl_mem), &ocl.speeds0);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 10, sizeof(cl_mem), &ocl.speeds1);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 11, sizeof(cl_mem), &ocl.speeds2);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 12, sizeof(cl_mem), &ocl.speeds3);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 13, sizeof(cl_mem), &ocl.speeds4);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 14, sizeof(cl_mem), &ocl.speeds5);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 15, sizeof(cl_mem), &ocl.speeds6);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 16, sizeof(cl_mem), &ocl.speeds7);
        checkError(err, "setting propagate arg 1", __LINE__);
        err = clSetKernelArg(ocl.propagate, 17, sizeof(cl_mem), &ocl.speeds8);
        checkError(err, "setting propagate arg 1", __LINE__);
      }
      err = clSetKernelArg(ocl.propagate, 26, sizeof(cl_int), &tt);
      checkError(err, "setting propagate arg 10", __LINE__);
    //argiements set above


      // Enqueue kernel
      size_t global[2] = {params.nx, params.ny};
      size_t local[2] = {LOCALSIZE,LOCALSIZE2};
      err = clEnqueueNDRangeKernel(ocl.queue, ocl.propagate,
                                   2, NULL, global, local, 0, NULL, NULL);
      checkError(err, "enqueueing propagate kernel", __LINE__);
  }

  float * tot_up = _mm_malloc((params.nx/LOCALSIZE)*(params.ny/LOCALSIZE2)*sizeof(float)*params.maxIters,64);
  int * tot_cellsp = _mm_malloc((params.nx/LOCALSIZE)*(params.ny/LOCALSIZE2)*sizeof(int)*params.maxIters,64);

  err = clFinish(ocl.queue);
  checkError(err, "writing for kernels to finish", __LINE__);

  err = clEnqueueReadBuffer(
    ocl.queue, ocl.partial_sum, CL_FALSE, 0,
    (params.nx/LOCALSIZE)*(params.ny/LOCALSIZE2)*sizeof(float)*params.maxIters, tot_up, 0, NULL, NULL);
  checkError(err, "reading velo data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.partial_sum2, CL_FALSE, 0,
    (params.nx/LOCALSIZE)*(params.ny/LOCALSIZE2)*sizeof(int)*params.maxIters, tot_cellsp, 0, NULL, NULL);
  checkError(err, "reading velo2 data", __LINE__);
  err = clFinish(ocl.queue);
  checkError(err, "writing for reduction to come back", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.tmp_speeds0, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s0, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.tmp_speeds1, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s1, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.tmp_speeds2, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s2, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.tmp_speeds3, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s3, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.tmp_speeds4, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s4, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.tmp_speeds5, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s5, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.tmp_speeds6, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s6, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.tmp_speeds7, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s7, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.tmp_speeds8, CL_FALSE, 0,
    sizeof(float) * (params.ny * params.nx), speeds.s8, 0, NULL, NULL);
  checkError(err, "writing speed data", __LINE__);
  float tot_u = 0;
  int tot_cells = 0;
  for (int tt = 0; tt < params.maxIters; tt++){
    tot_u = 0;
    tot_cells = 0;
    for(int i = 0; i < params.nx/LOCALSIZE*params.ny/LOCALSIZE2; i++){
      tot_u += tot_up[i+tt*params.nx/LOCALSIZE*params.ny/LOCALSIZE2];
      tot_cells += tot_cellsp[i+tt*params.nx/LOCALSIZE*params.ny/LOCALSIZE2];
    }
    av_vels[tt] = tot_u/tot_cells;

  }
  // Wait for reads to finish
  err = clFinish(ocl.queue);
  checkError(err, "waiting for propagate kernel", __LINE__);

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* loop over _all_ cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
    cells[ii + jj*params.nx].speeds[0] = speeds.s0[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[1] = speeds.s1[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[2] = speeds.s2[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[3] = speeds.s3[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[4] = speeds.s4[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[5] = speeds.s5[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[6] = speeds.s6[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[7] = speeds.s7[ii + jj*params.nx];
    cells[ii + jj*params.nx].speeds[8] = speeds.s8[ii + jj*params.nx];
    }
  }


  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, ocl));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}


float av_velocity(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl *ocl){
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }
  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  cl_int err;
  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "-cl-fast-relaxed-math", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->propagate = clCreateKernel(ocl->program, "propagate", &err);
  checkError(err, "creating propagate kernel", __LINE__);

  // Allocate OpenCL buffers
  ocl->speeds0 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->speeds1 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->speeds2 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->speeds3 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->speeds4 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->speeds5 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->speeds6 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->speeds7 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->speeds8 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_speeds0 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_speeds1 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_speeds2 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_speeds3 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_speeds4 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_speeds5 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_speeds6 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_speeds7 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_speeds8 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    params->nx * params->ny * sizeof(float), NULL, &err);
  checkError(err, "creating cells buffer", __LINE__);
  ocl->partial_sum = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    (params->nx/LOCALSIZE)*(params->ny/LOCALSIZE2)*sizeof(float)*params->maxIters, NULL, &err);
  checkError(err, "creating partial buffer", __LINE__);
  ocl->partial_sum2 = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    (params->nx/LOCALSIZE)*(params->ny/LOCALSIZE2)*sizeof(int)*params->maxIters, NULL, &err);
  checkError(err, "creating partial2 buffer", __LINE__);

  ocl->obstacles = clCreateBuffer(
    ocl->context, CL_MEM_READ_ONLY,
    sizeof(cl_int) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;


  clReleaseMemObject(ocl.speeds0);
  clReleaseMemObject(ocl.speeds1);
  clReleaseMemObject(ocl.speeds2);
  clReleaseMemObject(ocl.speeds3);
  clReleaseMemObject(ocl.speeds4);
  clReleaseMemObject(ocl.speeds5);
  clReleaseMemObject(ocl.speeds6);
  clReleaseMemObject(ocl.speeds7);
  clReleaseMemObject(ocl.speeds8);
  clReleaseMemObject(ocl.tmp_speeds0);
  clReleaseMemObject(ocl.tmp_speeds1);
  clReleaseMemObject(ocl.tmp_speeds2);
  clReleaseMemObject(ocl.tmp_speeds3);
  clReleaseMemObject(ocl.tmp_speeds4);
  clReleaseMemObject(ocl.tmp_speeds5);
  clReleaseMemObject(ocl.tmp_speeds6);
  clReleaseMemObject(ocl.tmp_speeds7);
  clReleaseMemObject(ocl.tmp_speeds8);
  clReleaseMemObject(ocl.partial_sum);
  clReleaseMemObject(ocl.partial_sum2);

  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.propagate);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles, ocl) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void checkError(cl_int err, const char *op, const int line)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice()
{
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[20];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(20, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);
  printf("num of platforms %d\n",num_platforms);
  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++)
  {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++)
  {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env)
  {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices)
  {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

return devices[device_index];
}
