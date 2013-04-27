// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!


#include <xmmintrin.h>


#define min(a,b) (((a)<(b))?(a):(b))
#define MAX_BLOCK 32

static void matmul_blocking(int N, const double* A, const double* B, double* restrict C, int BLOCK);
inline static void matmul_2x2(int N, const double* A, const double* B, double* restrict C);
inline static void matmul_sse(int Stride, int N, const double* A, const double* B, double* restrict C);

void matmul(int N, const double* A, const double* B, double* restrict C) {
  if(N==2)
    return matmul_2x2(2, A, B, C);
  else if(N<=MAX_BLOCK)
    return matmul_sse(0, N, A, B, C);
  else {
    return matmul_blocking(N,A,B,C,min(N,MAX_BLOCK));
  }

  /*  int i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < N; k++)
        C[i*N + j] += A[i*N+k]*B[k*N+j];
  */
}

void matmul_blocking(int N, const double* A, const double* B, double* restrict C, int BLOCK) {
  int i, j, k, jj, kk;
  double temp;
  double temp_a[2];
   __m128d a, b, c, c1;

  for (jj = 0; jj < N; jj+=BLOCK){
    //printf("jj %d\n",jj);

    for (kk = 0; kk < N; kk+=BLOCK){
      //printf("  kk %d\n",kk);

      for (i=0;i<N;i+=2){
	//printf("    i %d\n",i);

	//printf("=== A[%d*%d+%d], B[%d*%d+%d], C[%d*%d+%d] \n", jj,N,kk,kk,N,i, jj,N, i);

	//matmul_sse(N, BLOCK, &A[jj*N+kk], &B[kk*N+i], &C[jj*N + i]);

	for(j=jj;j<=min(jj+BLOCK-1,N);j++){
	  temp = 0.;
	  //printf("## C[%d*%d+%d] ",j,N,i);
	  //printf(" (%p, %d) ", &C[j*N+i], ((long long)( &C[j*N+i]))%16);
	  c1 = _mm_load_pd(&C[j*N+i]);
	  //printf(" ## \n");

	  for ( k= kk; k <=min(kk+BLOCK-1,N); k++){
	    a  = _mm_set1_pd(A[j*(N)+k]);
	    b  = _mm_load_pd(&B[k*(N)+i]);
	    c  = _mm_mul_pd(a, b);
	    c1 = _mm_add_pd(c, c1);

	    //temp += A[j*N+k]*B[k*N+i];
	  }
	  //C[j*N + i] += temp;
	  _mm_store_pd(&C[j*(N)+i], c1);
	}


	/*
   for(int i =0; i<N;i++){
     for (int j = 0; j < N; j+=2){
       c1 = _mm_load_pd(&C[i*(N+Stride)+j]);

       for(int k=0;k<N;k++){
	 a  = _mm_set1_pd(A[i*(N+Stride)+k]);
	 b  = _mm_load_pd(&B[k*(N+Stride)+j]);
	 c  = _mm_mul_pd(a, b);
	 c1 = _mm_add_pd(c, c1);
       }
       _mm_store_pd(&C[i*(N+Stride)+j], c1);
     }
   }*/


	/*for(j=jj;j<=min(jj+BLOCK-1,N);j++){
	  temp = 0.;
	  for ( k= kk; k <=min(kk+BLOCK-1,N); k++){
	    temp += A[j*N+k]*B[k*N+i];
	  }
	  C[j*N + i] += temp;
	}*/

      }
    }
  }
}

//Unrolled 2x2 Matrix Mult
void matmul_2x2_2(int N, const double* A, const double* B, double* restrict C) {
  C[2*0+0] += A[2*0+0]*B[2*0+0] + A[2*0+1]*B[2*1+0] ;
  C[2*0+1] += A[2*0+0]*B[2*0+1] + A[2*0+1]*B[2*1+1] ;
  C[2*1+0] += A[2*1+0]*B[2*0+0] + A[2*1+1]*B[2*1+0] ;
  C[2*1+1] += A[2*1+0]*B[2*0+1] + A[2*1+1]*B[2*1+1] ;
}

//Unrolled 2x2 Matrix Mult with SSE
inline static void matmul_2x2(int N, const double* A, const double* B, double* restrict C) {
  __m128d a0, a, a1, b, b1, c, c1, c2;

   //a = _mm_set1_pd(A[0]);
	 a0  = _mm_load_pd(&A[0]);
	 a = _mm_unpacklo_pd(a0,a0);
  b = _mm_load_pd(&B[0]);
  c = _mm_mul_pd(a, b);

  //a1 = _mm_set1_pd(A[1]);
	 a1 = _mm_unpackhi_pd(a0,a0);
  b1 = _mm_load_pd(&B[2*1+0]);
  c1 = _mm_mul_pd(a1, b1);

  a = _mm_load_pd(&C[0]);
  c2 = _mm_add_pd(c, c1);
  c1 = _mm_add_pd(a, c2);
  _mm_store_pd(&C[0], c1);


  //a = _mm_set1_pd(A[2*1+0]);
	 a0  = _mm_load_pd(&A[2*1+0]);
	 a = _mm_unpacklo_pd(a0,a0);
  c = _mm_mul_pd(a, b);

  a1 = _mm_set1_pd(A[2*1+1]);
  c1 = _mm_mul_pd(a1, b1);

  //a = _mm_load_pd(&C[2*1+0]);
	 a1 = _mm_unpackhi_pd(a0,a0);
  c2 = _mm_add_pd(c, c1);
  c1 = _mm_add_pd(a, c2);
  _mm_store_pd(&C[2*1+0], c1);
}

//N has to be even
//  for (i = 0; i < N; i++)
//    for (j = 0; j < N; j++)
//      for (k = 0; k < N; k++)
//       C[i*N + j] += A[i*N+k]*B[k*N+j];


inline static void matmul_sse(int Stride, int N, const double* A, const double* B, double* restrict C) {
  __m128d a, a1, a2, b, c, c1;
   for(int i =0; i<N;i++){
     for (int j = 0; j < N; j+=2){
       c1 = _mm_load_pd(&C[i*(N+Stride)+j]);

       for(int k=0;k<N;k+=2){
	 //a  = _mm_set1_pd(A[i*(N+Stride)+k]);
	 a  = _mm_load_pd(&A[i*(N+Stride)+k]);
	 a1 = _mm_unpacklo_pd(a,a);
	 a2 = _mm_unpackhi_pd(a,a);

	 b  = _mm_load_pd(&B[k*(N+Stride)+j]);
	 c  = _mm_mul_pd(a1, b);
	 c1 = _mm_add_pd(c, c1);

	 b  = _mm_load_pd(&B[(k+1)*(N+Stride)+j]);
	 c  = _mm_mul_pd(a2, b);
	 c1 = _mm_add_pd(c, c1);

       }
       _mm_store_pd(&C[i*(N+Stride)+j], c1);
     }
   }

}

/*
inline static void matmul_sse(int N, const double* A, const double* B, double* restrict C) {
   __m128d a, b, c, c1;
   for(int i =0; i<N;i++){
     for (int j = 0; j < N; j+=2){
       c1 = _mm_load_pd(&C[i*N+j]);

       for(int k=0;k<N;k++){
	 a  = _mm_set1_pd(A[i*N+k]);
	 b  = _mm_load_pd(&B[k*N+j]);
	 c  = _mm_mul_pd(a, b);
	 c1 = _mm_add_pd(c, c1);
       }
       _mm_store_pd(&C[i*N+j], c1);
     }
   }

}
*/
