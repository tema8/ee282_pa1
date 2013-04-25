// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!


#include <xmmintrin.h>


#define min(a,b) (((a)<(b))?(a):(b))
#define MAX_BLOCK 32

static void matmul_blocking(int N, const double* A, const double* B, double* C, int BLOCK);

void matmul(int N, const double* A, const double* B, double* C) {
  if(N>2) {
    matmul_blocking(N,A,B,C,min(N,MAX_BLOCK));
    return;
  }

  int i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < N; k++)
        C[i*N + j] += A[i*N+k]*B[k*N+j];

}

void matmul_blocking(int N, const double* A, const double* B, double* C, int BLOCK) {
  int i, j, k, jj, kk;
  double temp;
  double temp_a[2];
  //__m128d a, b, c;

  for (jj = 0; jj < N; jj+=BLOCK){
    //printf("jj %d\n",jj);

    for (kk = 0; kk < N; kk+=BLOCK){
      //printf("  kk %d\n",kk);
      for (i=0;i<N;i++){
	//printf("    i %d\n",i);
	
	for(j=jj;j<=min(jj+BLOCK-1,N);j++){
	  temp = 0.;
	  for ( k= kk; k <=min(kk+BLOCK-1,N); k+=1){

	    //a = _mm_load_pd(&A[j*N+k]);
	    //b = _mm_load_pd(&B[i*N+k]);
	    //c = _mm_mul_pd(a, b);
	    //_mm_store_pd(temp_a, c);
	    //temp+=temp_a[0];
	    //temp+=temp_a[1];

	    //temp += A[j*N+k]*B[i*N+k];
	    temp += A[j*N+k]*B[k*N+i];
	    //temp += A[j*N+k+1]*B[i*N+k+1];
	  }
	  C[j*N + i] += temp;
	  //printf("C[%d][%d]\n",j,i);
	}
      }
    }
  }
}

