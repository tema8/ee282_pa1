// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!


#include <xmmintrin.h>

#define MAX_BLOCK 32
//getconf LEVEL1_DCACHE_LINESIZE
#define CACHE_LINE (64/sizeof(double))

static void matmul_blocking(int N, const double* A, const double* B, double * restrict C);
inline static void matmul_sse(int N, const double* A, const double* B, double * restrict C);
inline static void matmul_2x2(const double* A, const double* B, double* restrict C);
inline static void matmul_4x4 ( const double *A, const double *B, double * restrict C);
inline static void matmul_8x8 ( const double *A, const double *B, double * restrict C);
inline static void matmul_16x16 ( const double *A, const double *B, double * restrict C);
inline static void matmul_32x32 ( const double *A, const double *B, double * restrict C);

void matmul(int N, const double* A, const double* B, double* restrict C) {
  if(N==2)
    return matmul_2x2( A, B, C);
  else if(N==4)
    return matmul_4x4( A, B, C);
  else if(N==8)
    return matmul_8x8( A, B, C);
  else if(N==16)
    return matmul_16x16( A, B, C);
  else if(N==32)
    return matmul_32x32( A, B, C);
  else if(N<=MAX_BLOCK)
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






inline static void matmul_2x2(const double* A, const double* B, double* restrict C) {


  const double *in1, *in2;
  double * restrict res;


	res = C;
	in1 = A;

	//for(int t = 0;t <2;t++){
	  in2 = B;

	    __m128d a = _mm_load_sd(in1);
	    a = _mm_unpacklo_pd(a,a);

	    __m128d b  = _mm_load_pd(in2);
	    __m128d c  = _mm_load_pd(res);
	    __m128d c1 = _mm_add_pd(c, _mm_mul_pd(a, b));
	    _mm_store_pd(res, c1);

	    __m128d a1 = _mm_load_sd(in1+1);
	    a1 = _mm_unpacklo_pd(a1,a1);

	    __m128d b2  = _mm_load_pd(in2+2);

	    c1=  _mm_add_pd(c1, _mm_mul_pd(a1, b2));
	    _mm_store_pd(res, c1);

	    in1+=2;
	    in2+=2;
	    res+=2;


	    a = _mm_load_sd(in1);
	    a = _mm_unpacklo_pd(a,a);

	    //b  = _mm_load_pd(in2);
	    c  = _mm_load_pd(res);
	    c1 = _mm_add_pd(c, _mm_mul_pd(a, b));
	    _mm_store_pd(res, c1);

	    a1 = _mm_load_sd(in1+1);
	    a1 = _mm_unpacklo_pd(a1,a1);

	    //b2  = _mm_load_pd(in2+2);

	    c1=  _mm_add_pd(c1, _mm_mul_pd(a1, b2));
	    _mm_store_pd(res, c1);



	    //}
}



inline static void matmul_4x4 ( const double *A, const double *B, double * restrict C) {
  const double *in1, *in2;
  double * restrict res;

  res = C;
  in1 = A;

  for(int t = 0;t <4;t++){
    in2 = B;

    for(int l = 0; l<4;l++  ){
      __m128d a = _mm_load_sd(&in1[l]);
      a = _mm_unpacklo_pd(a,a);

      for(int m = 0; m<4;m+=2){
	
	__m128d b  = _mm_load_pd(&in2[m]);
	__m128d c  = _mm_load_pd(&res[m]);
	c =  _mm_add_pd(c, _mm_mul_pd(a, b));
	_mm_store_pd(&res[m], c);
      }

      in2+=4;
    }

    res+=4;
    in1+=4;
  }
}



inline static void matmul_8x8 ( const double *A, const double *B, double * restrict C) {
  const double *in1, *in2;
  double * restrict res;

  res = C;
  in1 = A;

  for(int t = 0;t <8;t++){
    in2 = B;

    for(int l = 0; l<8;l++  ){
      __m128d a = _mm_load_sd(&in1[l]);
      a = _mm_unpacklo_pd(a,a);

      for(int m = 0; m<8;m+=2){
	
	__m128d b  = _mm_load_pd(&in2[m]);
	__m128d c  = _mm_load_pd(&res[m]);
	c =  _mm_add_pd(c, _mm_mul_pd(a, b));
	_mm_store_pd(&res[m], c);
      }

      in2+=8;
    }

    res+=8;
    in1+=8;
  }
}



inline static void matmul_16x16 ( const double *A, const double *B, double * restrict C) {
  const double *in1, *in2;
  double * restrict res;

  res = C;
  in1 = A;

  for(int t = 0;t <16;t++){
    in2 = B;

    for(int l = 0; l<16;l++  ){
      __m128d a = _mm_load_sd(&in1[l]);
      a = _mm_unpacklo_pd(a,a);

      for(int m = 0; m<16;m+=2){
	
	__m128d b  = _mm_load_pd(&in2[m]);
	__m128d c  = _mm_load_pd(&res[m]);
	c =  _mm_add_pd(c, _mm_mul_pd(a, b));
	_mm_store_pd(&res[m], c);
      }

      in2+=16;
    }

    res+=16;
    in1+=16;
  }
}


inline static void matmul_32x32 ( const double *A, const double *B, double * restrict C) {
  const double *in1, *in2;
  double * restrict res;

  res = C;
  in1 = A;

  for(int t = 0;t <32;t++){
    in2 = B;

    for(int l = 0; l<32;l++  ){
      __m128d a = _mm_load_sd(&in1[l]);
      a = _mm_unpacklo_pd(a,a);

      for(int m = 0; m<32;m+=2){
	
	__m128d b  = _mm_load_pd(&in2[m]);
	__m128d c  = _mm_load_pd(&res[m]);
	c =  _mm_add_pd(c, _mm_mul_pd(a, b));
	_mm_store_pd(&res[m], c);
      }

      in2+=32;
    }

    res+=32;
    in1+=32;
  }
}













//N has to be even
inline static void matmul_sse(int N, const double* A, const double* B, double* restrict C) {


  const double *in1, *in2;
  double * restrict res;


	res = C;
	in1 = A;

	for(int t = 0;t <N;t++){
	  in2 = B;

	  for(int l = 0; l<N;l++  ){
	    __m128d a = _mm_load_sd(&in1[l]);
	    a = _mm_unpacklo_pd(a,a);

	    for(int m = 0; m<N;m+=2){
	    __m128d b  = _mm_load_pd(&in2[m]);
	    __m128d c  = _mm_load_pd(&res[m]);
	    c=  _mm_add_pd(c, _mm_mul_pd(a, b));
	    _mm_store_pd(&res[m], c);

	    }
	    in2+=N;
	  }
	  res+=N;
	  in1+=N;
	}

  /*
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
  */
}


