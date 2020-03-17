#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

kernel void propagate(global float* restrict speeds0, global float* restrict speeds1, global float* restrict speeds2, global float* restrict speeds3, global float* restrict speeds4, global float* restrict speeds5, global float* restrict speeds6,
  global float* restrict speeds7, global float* restrict speeds8, global float* restrict tmp_speeds0, global float* restrict tmp_speeds1, global float* restrict tmp_speeds2, global float* restrict tmp_speeds3, global float* restrict tmp_speeds4,
  global float* restrict tmp_speeds5, global float* restrict tmp_speeds6, global float* restrict tmp_speeds7, global float* restrict tmp_speeds8, global int* restrict obstacles, int nx, int ny, float omega, local float* local_sum, local int* local_sum2,
  global float* partial_sum, global int* partial_sum2, int iters,float densityaccel){

  /* get column and row indices */
  const int ii = get_global_id(0);
  const int jj = get_global_id(1);

  const float c_sq_inv = 3.f;
  const float c_sq = half_recip(c_sq_inv); /* square of speed of sound */
  const float temp1 = 4.5f;
  const float w1 = half_recip(9.f);
  const float w0 = 4.f * w1;  /* weighting factor */
  const float w2 = half_recip(36.f); /* weighting factor */
  const float w11 = densityaccel * w1;
  const float w21 = densityaccel * w2;

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  const int y_n = (jj + 1) % ny;
  const int x_e = (ii + 1) % nx;
  const int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  const int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */

  float tmp_s0 = speeds0[ii + jj*nx];
  float tmp_s1 = (jj == ny-2 && (!obstacles[x_w + jj*nx] && isgreater((speeds3[x_w + jj*nx] - w11) , 0.f) && isgreater((speeds6[x_w + jj*nx] - w21) , 0.f) && isgreater((speeds7[x_w + jj*nx] - w21) , 0.f))) ? speeds1[x_w + jj*nx]+w11 : speeds1[x_w + jj*nx];
  float tmp_s2 = speeds2[ii + y_s*nx];
  float tmp_s3 = (jj == ny-2 && (!obstacles[x_e + jj*nx] && isgreater((speeds3[x_e + jj*nx] - w11) , 0.f) && isgreater((speeds6[x_e + jj*nx] - w21) , 0.f) && isgreater((speeds7[x_e + jj*nx] - w21) , 0.f))) ? speeds3[x_e + jj*nx]-w11 : speeds3[x_e + jj*nx];
  float tmp_s4 = speeds4[ii + y_n*nx];
  float tmp_s5 = (y_s == ny-2 && (!obstacles[x_w + y_s*nx] && isgreater((speeds3[x_w + y_s*nx] - w11) , 0.f) && isgreater((speeds6[x_w + y_s*nx] - w21) , 0.f) && isgreater((speeds7[x_w + y_s*nx] - w21) , 0.f))) ? speeds5[x_w + y_s*nx]+w21 : speeds5[x_w + y_s*nx];
  float tmp_s6 = (y_s == ny-2 && (!obstacles[x_e + y_s*nx] && isgreater((speeds3[x_e + y_s*nx] - w11) , 0.f) && isgreater((speeds6[x_e + y_s*nx] - w21) , 0.f) && isgreater((speeds7[x_e + y_s*nx] - w21) , 0.f))) ? speeds6[x_e + y_s*nx]-w21 : speeds6[x_e + y_s*nx];
  float tmp_s7 = (y_n == ny-2 && (!obstacles[x_e + y_n*nx] && isgreater((speeds3[x_e + y_n*nx] - w11) , 0.f) && isgreater((speeds6[x_e + y_n*nx] - w21) , 0.f) && isgreater((speeds7[x_e + y_n*nx] - w21) , 0.f))) ? speeds7[x_e + y_n*nx]-w21 : speeds7[x_e + y_n*nx];
  float tmp_s8 = (y_n == ny-2 && (!obstacles[x_w + y_n*nx] && isgreater((speeds3[x_w + y_n*nx] - w11) , 0.f) && isgreater((speeds6[x_w + y_n*nx] - w21) , 0.f) && isgreater((speeds7[x_w + y_n*nx] - w21) , 0.f))) ? speeds8[x_w + y_n*nx]+w21 : speeds8[x_w + y_n*nx];

  /* compute local density total */
  float local_density = tmp_s0 + tmp_s1 + tmp_s2 + tmp_s3 + tmp_s4  + tmp_s5  + tmp_s6  + tmp_s7  + tmp_s8;
  const float local_density_recip = half_recip(local_density);
  /* compute x velocity component */
  float u_x = (tmp_s1
                + tmp_s5
                + tmp_s8
                - tmp_s3
                - tmp_s6
                - tmp_s7)
               * local_density_recip;
  /* compute y velocity component */
  float u_y = (tmp_s2
                + tmp_s5
                + tmp_s6
                - tmp_s4
                - tmp_s7
                - tmp_s8)
               * local_density_recip;

  /* velocity squared */
  const float temp2 = - (u_x * u_x + u_y * u_y)* half_recip((2.f * c_sq));

  /* equilibrium densities */
  float d_equ[NSPEEDS];
  /* zero velocity density: weight w0 */
  d_equ[0] = w0 * local_density
             * (1.f + temp2);
  /* axis speeds: weight w1 */
  d_equ[1] = w1 * local_density * (1.f + u_x * c_sq_inv
                                   + (u_x * u_x) * temp1
                                   + temp2);
  d_equ[2] = w1 * local_density * (1.f + u_y * c_sq_inv
                                   + (u_y * u_y) * temp1
                                   + temp2);
  d_equ[3] = w1 * local_density * (1.f - u_x * c_sq_inv
                                   + (u_x * u_x) * temp1
                                   + temp2);
  d_equ[4] = w1 * local_density * (1.f - u_y * c_sq_inv
                                   + (u_y * u_y) * temp1
                                   + temp2);
  /* diagonal speeds: weight w2 */
  d_equ[5] = w2 * local_density * (1.f + (u_x + u_y) * c_sq_inv
                                   + ((u_x + u_y) * (u_x + u_y)) * temp1
                                   + temp2);
  d_equ[6] = w2 * local_density * (1.f + (-u_x + u_y) * c_sq_inv
                                   + ((-u_x + u_y) * (-u_x + u_y)) * temp1
                                   + temp2);
  d_equ[7] = w2 * local_density * (1.f + (-u_x - u_y) * c_sq_inv
                                   + ((-u_x - u_y) * (-u_x - u_y)) * temp1
                                   + temp2);
  d_equ[8] = w2 * local_density * (1.f + (u_x - u_y) * c_sq_inv
                                   + ((u_x - u_y) * (u_x - u_y)) * temp1
                                   + temp2);

  float tmp;
  int expression = obstacles[ii + jj*nx];
  tmp_s0 = select((tmp_s0 + omega * (d_equ[0] - tmp_s0)),tmp_s0,expression);
  tmp = tmp_s1;
  tmp_s1 = select((tmp_s1 + omega * (d_equ[1] - tmp_s1)),tmp_s3,expression);
  tmp_s3 = select((tmp_s3 + omega * (d_equ[3] - tmp_s3)),tmp,expression);
  tmp = tmp_s2;
  tmp_s2 = select((tmp_s2 + omega * (d_equ[2] - tmp_s2)),tmp_s4,expression);
  tmp_s4 = select((tmp_s4 + omega * (d_equ[4] - tmp_s4)),tmp,expression);
  tmp = tmp_s5;
  tmp_s5 = select((tmp_s5 + omega * (d_equ[5] - tmp_s5)),tmp_s7,expression);
  tmp_s7 = select((tmp_s7 + omega * (d_equ[7] - tmp_s7)),tmp,expression);
  tmp = tmp_s6;
  tmp_s6 = select((tmp_s6 + omega * (d_equ[6] - tmp_s6)),tmp_s8,expression);
  tmp_s8 = select((tmp_s8 + omega * (d_equ[8] - tmp_s8)),tmp,expression);

  /* local density total */
  local_density =  half_recip(tmp_s0 + tmp_s1 + tmp_s2 + tmp_s3 + tmp_s4 + tmp_s5 + tmp_s6 + tmp_s7 + tmp_s8);

  /* x-component of velocity */
  u_x = (tmp_s1
                + tmp_s5
                + tmp_s8
                - tmp_s3
                - tmp_s6
                - tmp_s7)
               * local_density;
  /* compute y velocity component */
  u_y = (tmp_s2
                + tmp_s5
                + tmp_s6
                - tmp_s4
                - tmp_s7
                - tmp_s8)
               * local_density;


  tmp_speeds0[ii + jj*nx] = tmp_s0;
  tmp_speeds1[ii + jj*nx] = tmp_s1;
  tmp_speeds2[ii + jj*nx] = tmp_s2;
  tmp_speeds3[ii + jj*nx] = tmp_s3;
  tmp_speeds4[ii + jj*nx] = tmp_s4;
  tmp_speeds5[ii + jj*nx] = tmp_s5;
  tmp_speeds6[ii + jj*nx] = tmp_s6;
  tmp_speeds7[ii + jj*nx] = tmp_s7;
  tmp_speeds8[ii + jj*nx] = tmp_s8;


  int local_idi = get_local_id(0);
  int local_idj = get_local_id(1);
  int local_sizei = get_local_size(0);
  int local_sizej = get_local_size(1);
  /* accumulate the norm of x- and y- velocity components */
  local_sum[local_idi + local_idj*local_sizei] = (obstacles[ii + jj*nx]) ? 0 : hypot(u_x,u_y);
  /* increase counter of inspected cells */
  local_sum2[local_idi + local_idj*local_sizei] = (obstacles[ii + jj*nx]) ? 0 : 1 ;
  barrier(CLK_LOCAL_MEM_FENCE);
  int group_id = get_group_id(0);
  int group_size = get_num_groups(0);
  int group_size2 = get_num_groups(1);
  int group_id2 = get_group_id(1);
  if(local_idi == 0 && local_idj == 0){
    float sum = 0.0f;
    int sum2 = 0;
    for(int i = 0; i<local_sizei*local_sizej; i++){
      sum += local_sum[i];
      sum2 += local_sum2[i];
    }
    partial_sum[group_id+group_id2*group_size+iters*group_size*group_size2] = sum;
    partial_sum2[group_id+group_id2*group_size+iters*group_size*group_size2] = sum2;
  }

}
