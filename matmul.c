// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!


#include <xmmintrin.h>

#define MAX_BLOCK 32
//getconf LEVEL1_DCACHE_LINESIZE
#define CACHE_LINE (64/sizeof(double))

static void matmul_blocking(int N, const double* A, const double* B, double * restrict C);
inline static void matmul_sse(int N, const double* A, const double* B, double * restrict C);

void matmul(int N, const double* A, const double* B, double* restrict C) {
  if(N<=MAX_BLOCK)
    return matmul_sse(N, A, B, C);
  else {
    return matmul_blocking(N,A,B,C);
  }

  /*  int i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < N; k++)
        C[i*N + j] += A[i*N+k]*B[k*N+j];
  */
}


void matmul_blocking(int N, const double *A, const double *B, double * restrict C) {
  const double *in1, *in2;
  double * restrict res;

  for(int i = 0; i < N; i+=CACHE_LINE) {
    for (int j = 0; j < N; j+=CACHE_LINE) {
      for (int k = 0; k < N; k+=CACHE_LINE) {
	res = &C[i*N+j];
	in1 = &A[(i)*N+k];
	for(int t = 0;t <CACHE_LINE;t++){
	  in2 = &B[(k)*N+j];
	  for(int l = 0; l<CACHE_LINE;l++  ){
	    __m128d a = _mm_load_sd(&in1[l]);
	    a = _mm_unpacklo_pd(a,a);

	    for(int m = 0; m<CACHE_LINE;m+=2){
	    //C[i*N+j+l] += A[(i)*N+k+t] * B[(k+t)*N+j+l];
	    //res[m] += in1[l] * in2[m];
	    __m128d b  = _mm_load_pd(&in2[m]);
	    __m128d c  = _mm_load_pd(&res[m]);
	    c=  _mm_add_pd(c, _mm_mul_pd(a, b));
	    _mm_store_pd(&res[m], c);

	    //printf("%d,%d   C[%d][%d] +=  A[%d][%d]*B[%d][%d]     ",t,l  , i, j+l,    i, k+t, k+t, j+l);
	    }
	    in2+=N;
	  }
	  res+=N;
	  in1+=N;
	}

      }
    }
  }
}


//N has to be even
inline static void matmul_sse(int N, const double* A, const double* B, double* restrict C) {
  __m128d a, a1, a2, b, c, c1;
   for(int i =0; i<N;i++){
     for (int j = 0; j < N; j+=2){
       c1 = _mm_load_pd(&C[i*N+j]);

       for(int k=0;k<N;k+=2){
	 //a  = _mm_set1_pd(A[i*N+k]);
	 a  = _mm_load_pd(&A[i*N+k]);
	 a1 = _mm_unpacklo_pd(a,a);
	 a2 = _mm_unpackhi_pd(a,a);

	 b  = _mm_load_pd(&B[k*N+j]);
	 c  = _mm_mul_pd(a1, b);
	 c1 = _mm_add_pd(c, c1);

	 b  = _mm_load_pd(&B[(k+1)*N+j]);
	 c  = _mm_mul_pd(a2, b);
	 c1 = _mm_add_pd(c, c1);

       }
       _mm_store_pd(&C[i*N+j], c1);
     }
   }

}


