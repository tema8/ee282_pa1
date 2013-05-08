// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!


#include <xmmintrin.h>
#include <stdio.h>

#define MAX_BLOCK 32
//getconf LEVEL1_DCACHE_LINESIZE
//#define CACHE_LINE (64/sizeof(double))
#define CACHE_LINE 32

static void matmul_blocking(int N, const double* A, const double* B, double * restrict C);
inline static void matmul_sse(int N, const double* A, const double* B, double * restrict C);
inline static void matmul_2x2(const double* A, const double* B, double* restrict C);
inline static void matmul_4x4 ( const double *A, const double *B, double * restrict C);
inline static void matmul_8x8 ( const double *A, const double *B, double * restrict C);
inline static void matmul_16x16 ( const double *A, const double *B, double * restrict C);
inline static void matmul_32x32 ( const double *A, const double *B, double * restrict C);
//inline static void matmul_64x64( const double *A, const double *B, double * restrict C) ;

inline static void matmul_32x32_stride ( const double *A, int Stride_A, const double *B, int Stride_B, double * restrict C, int Stride_C);
static void matmul_strasen(int N, const double* A, int Stride_A, const double* B, int Stride_B, double* restrict C, int Stride_C);

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
    //return matmul_32x32_stride( A, 32, B, 32, C, 32);
    return matmul_32x32( A, B, C);
  else if(N>64){
    //printf("\n===Stransen===\n");
    return matmul_strasen(N, A, N, B, N, C, N);
  }
  //else if(N<=MAX_BLOCK)
  //  return matmul_sse(N, A, B, C);
  else {
    //return matmul_sse(N, A, B, C);
    return matmul_blocking(N,A,B,C);
  }

  /*  int i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < N; k++)
        C[i*N + j] += A[i*N+k]*B[k*N+j];
  */
}





void matadd(int size, const double *A, int Stride_A,const double *B, int Stride_B, double * restrict C, int Stride_C) {
  for(int i = 0; i < size; i++)
    for (int j = 0; j < size; j+=2){
       __m128d a = _mm_load_sd(&A[i*Stride_A+j]);
       __m128d b = _mm_load_sd(&B[i*Stride_B+j]);
       _mm_store_pd(&C[i*Stride_C+j], _mm_add_pd(a,b) );
       //C[i*Stride_C+j] = A[i*Stride_A+j] + B[i*Stride_B+j];
    }
}

void matacc(int size, const double *A, int Stride_A,const double *B, int Stride_B, double * restrict C, int Stride_C) {
  for(int i = 0; i < size; i++)
    for (int j = 0; j < size; j+=2){
       __m128d a = _mm_load_sd(&A[i*Stride_A+j]);
       __m128d b = _mm_load_sd(&B[i*Stride_B+j]);
       __m128d c = _mm_load_sd(&C[i*Stride_C+j]);
       _mm_store_pd(&C[i*Stride_C+j], _mm_add_pd(c,_mm_add_pd(a,b)) );
       //C[i*Stride_C+j] = A[i*Stride_A+j] + B[i*Stride_B+j];
    }
}



void matsub(int size, const double *A, int Stride_A,const double *B, int Stride_B, double * restrict C, int Stride_C){
  for(int i = 0; i < size; i++)
    for (int j = 0; j < size; j+=2){
       __m128d a = _mm_load_sd(&A[i*Stride_A+j]);
       __m128d b = _mm_load_sd(&B[i*Stride_B+j]);
       _mm_store_pd(&C[i*Stride_C+j], _mm_sub_pd(a,b) );
       //C[i*Stride_C+j] = A[i*Stride_A+j] - B[i*Stride_B+j];
    }
}


void matmul_strasen(int N, const double* A, int Stride_A, const double* B, int Stride_B, double* restrict C, int Stride_C) {
  if(N<=64)
    matmul_32x32_stride(A, Stride_A, B, Stride_B, C, Stride_C);
  else{
    double *M[9];

    for(int i=0;i<9;i++)
      M[i] = (double *) malloc(N/2*N/2*sizeof(double));


    matadd(N/2, A, Stride_A, &A[N/2*N+N/2], Stride_A, M[7], N/2);//A[1][1]+A[2][2]
    matadd(N/2, B, Stride_B, &B[N/2*N+N/2], Stride_B, M[8], N/2);//B[1][1]+B[2][2]
    matmul_strasen(N/2, M[7], N/2, M[8], N/2, M[0], N/2);

    matadd(N/2, &A[N/2*N+0], Stride_A , &A[N/2*N+N/2], Stride_A, M[7], N/2);//A[2][1]+A[2][2]
    matmul_strasen(N/2, M[7], N/2, B, Stride_B, M[1], N/2);

    matsub(N/2, &B[0*N+N/2], Stride_B , &B[N/2*N+N/2], Stride_B, M[8], N/2);//B[1][2]-B[2][2]
    matmul_strasen(N/2, A, Stride_A, M[8], N/2, M[2], N/2);


    matsub(N/2, &B[N/2*N+0], Stride_B , B, Stride_B, M[7], N/2);//B[2][1]-B[1][1]
    matmul_strasen(N/2, &A[N/2*N+N/2], Stride_A, M[7], N/2, M[3], N/2);

    matadd(N/2, A, Stride_A, &A[0*N+N/2], Stride_A, M[8],N/2);//A[1][1]+A[1][2]
    matmul_strasen(N/2, M[8], N/2, &B[N/2*N+N/2], Stride_B, M[4], N/2);

    matsub(N/2, &A[N/2*N+0], Stride_A , A, Stride_A, M[7], N/2);//A[2][1]-A[1][1]
    matadd(N/2, B, Stride_B, &B[0*N+N/2], Stride_B, M[8], N/2);//B[1][1]+B[1][2]
    matmul_strasen(N/2, M[7], N/2, M[8], N/2, M[5], N/2);

    matsub(N/2, &A[0*N+N/2], Stride_A , &A[N/2*N+N/2], Stride_A, M[7], N/2);//A[1][2]-A[2][2]
    matadd(N/2, &B[N/2*N+0], Stride_B , &B[N/2*N+N/2], Stride_B, M[8], N/2);//B[2][1]+B[2][2]
    matmul_strasen(N/2, M[7], N/2, M[8], N/2, M[6], N/2);

    matadd(N/2, M[0], N/2, M[3], N/2, M[7], N/2);
    matsub(N/2, M[6], N/2, M[4], N/2, M[8], N/2);
    matacc(N/2, M[7], N/2, M[8], N/2, C, Stride_C);
 
    matacc(N/2, M[2], N/2, M[4], N/2, &C[0*N+N/2], Stride_C);

    matacc(N/2, M[1], N/2, M[3], N/2, &C[N/2*N+0], Stride_C);

    matsub(N/2, M[0], N/2, M[1], N/2, M[7], N/2);
    matadd(N/2, M[2], N/2, M[5], N/2, M[8], N/2);
    matacc(N/2, M[7], N/2, M[8], N/2, &C[N/2*N+N/2], Stride_C);

    for(int i=0;i<9;i++){
      free(M[i]);
    }
  }
  
}




 





#define UNROLL 4

void matmul_blocking(int N, const double *A, const double *B, double * restrict C) {

  //const int UNROLL = 4;

  const double *in1, *in2;
  double * restrict res;
  __m128d a[UNROLL];
  __m128d b[UNROLL];
  __m128d temp[UNROLL/2];


  for(int i = 0; i < N; i+=CACHE_LINE) {
    for (int j = 0; j < N; j+=CACHE_LINE) {
      for (int k = 0; k < N; k+=CACHE_LINE) {
	res = &C[i*N+j];
	in1 = &A[(i)*N+k];



	//////////////////


  //t correspond to i
  for(int t = 0;t <CACHE_LINE;t++){
    in2 = &B[(k)*N+j];

    //l correspond to k
    for(int l = 0; l<CACHE_LINE;l+= UNROLL ){
      //set up
      for(int tt =0; tt< UNROLL; tt++)
	a[tt] =  _mm_load1_pd(&in1[l+tt]);

      //m corresponds to j
      for(int m = 0; m<CACHE_LINE;m+=2){
	//set up
	for(int tt =0; tt< UNROLL;tt++)
	  b[tt] =  _mm_load_pd(&in2[(tt)*N+m]);

	__m128d c   = _mm_load_pd(&res[m]);
	//set up
	for(int tt =0; tt< UNROLL;tt++)
	  b[tt] =  _mm_mul_pd(a[tt], b[tt]);
	//set up
	for(int tt =0; tt< UNROLL/2;tt++)
	  temp[tt] =  _mm_add_pd(b[2*tt], b[2*tt+1]);

	if(UNROLL == 1)
	  c =  _mm_add_pd(c, b[0] );
	else if(UNROLL == 2)
	  c =  _mm_add_pd(c, _mm_add_pd(b[0], b[1]) );
	else
	  for(int tt =0; tt< UNROLL/4;tt++)
	    c =  _mm_add_pd(c, _mm_add_pd(temp[2*tt], temp[2*tt+1]));

	_mm_store_pd(&res[m], c);
      }
	    in2+=N*UNROLL;
    }

	  res+=N;
	  in1+=N;

  }









	////////////////



      }
    }
  }
}






inline static void matmul_2x2(const double* A, const double* B, double* restrict C) {
  
  
  const double *in1, *in2;
  double * restrict res;

  res = C;
  in1 = A;

  in2 = B;

  //__m128d a = _mm_load_pd(in1);
  __m128d a1 = _mm_load1_pd(in1);//_mm_unpacklo_pd(a,a);

  __m128d b  = _mm_load_pd(in2);
  __m128d c  = _mm_load_pd(res);
  __m128d a  = _mm_load1_pd(in1+1);//_mm_unpackhi_pd(a,a);
  __m128d b2  = _mm_load_pd(in2+2);
  __m128d c1 = _mm_add_pd(c, _mm_mul_pd(a1, b));


  c1=  _mm_add_pd(c1, _mm_mul_pd(a, b2));
  _mm_store_pd(res, c1);

  //in1+=2;
  //in2+=2;
  res+=2;


  //a = _mm_load_pd(in1);
  a1 =  _mm_load1_pd(in1+2);//_mm_unpacklo_pd(a,a);

  c  = _mm_load_pd(res);
  a =  _mm_load1_pd(in1+3);//_mm_unpackhi_pd(a,a);

  c1 = _mm_add_pd(c, _mm_mul_pd(a1, b));

  c1=  _mm_add_pd(c1, _mm_mul_pd(a, b2));
  _mm_store_pd(res, c1);

}



inline static void matmul_4x4 ( const double *A, const double *B, double * restrict C) {
  const double *in1, *in2;
  double * restrict res;

  res = C;
  in1 = A;

  for(int t = 0;t <4;t++){
    in2 = B;

    for(int l = 0; l<4;l++  ){
      __m128d a = _mm_load1_pd(in1+l);// _mm_load_sd(&in1[l]);
      //a = _mm_unpacklo_pd(a,a);

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
      __m128d a = _mm_load1_pd(in1+l);//_mm_load_sd(&in1[l]);
      //a = _mm_unpacklo_pd(a,a);

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
      __m128d a = _mm_load1_pd(in1+l);//_mm_load_sd(&in1[l]);
      //a = _mm_unpacklo_pd(a,a);

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


//#define UNROLL 1

inline static void matmul_32x32 ( const double *A, const double *B, double * restrict C) {
  //const int UNROLL = 4;
  const double *in1, *in2;
  double * restrict res;

  const int N = 32;

  __m128d a[UNROLL];
  __m128d b[UNROLL];
  __m128d temp[UNROLL/2];

  res = C;
  in1 = A;
  in2 = B;

  
  //t correspond to i
  for(int i = 0;i <32;i++){
    //l correspond to k
    for(int k = 0; k<32;k+= UNROLL ){
      //set up
      for(int tt =0; tt< UNROLL; tt++)
	a[tt] =  _mm_load1_pd(&in1[i*32+k+tt]);

      //m corresponds to j
      for(int j = 0; j<32;j+=2){
	//set up
	for(int tt =0; tt< UNROLL;tt++)
	  b[tt] =  _mm_load_pd(&in2[(k+tt)*32+j]);

	__m128d c   = _mm_load_pd(&res[i*32+j]);
	//set up
	for(int tt =0; tt< UNROLL;tt++)
	  b[tt] =  _mm_mul_pd(a[tt], b[tt]);
	//set up
	for(int tt =0; tt< UNROLL/2;tt++)
	  temp[tt] =  _mm_add_pd(b[2*tt], b[2*tt+1]);

	if(UNROLL == 1)
	  c =  _mm_add_pd(c, b[0] );
	else if(UNROLL == 2)
	  c =  _mm_add_pd(c, _mm_add_pd(b[0], b[1]) );
	else
	  for(int tt =0; tt< UNROLL/4;tt++)
	    c =  _mm_add_pd(c, _mm_add_pd(temp[2*tt], temp[2*tt+1]));

	_mm_store_pd(&res[i*32+j], c);
      }
    }

  }

}

inline static void matmul_32x32_stride_ ( const double *A, int Stride_A, const double *B, int Stride_B, double * restrict C, int Stride_C) {
  const double *in1, *in2;
  double * restrict res;
  const int N = 32;

  res = C;
  in1 = A;

  for(int t = 0;t < N;t++){
    in2 = B;

    for(int l = 0; l<N;l++ ){
      __m128d a = _mm_load1_pd(&in1[l]);

      for(int m = 0; m<N;m+=2){

__m128d b = _mm_load_pd(&in2[m]);
__m128d c = _mm_load_pd(&res[m]);
c = _mm_add_pd(c, _mm_mul_pd(a, b));
_mm_store_pd(&res[m], c);
      }

      in2+=Stride_B;
    }

    res+=Stride_C;
    in1+=Stride_A;
  }
}

//#define UNROLL 1

inline static void matmul_32x32_stride ( const double *A, int Stride_A, const double *B, int Stride_B, double * restrict C, int Stride_C){
  const double *in1, *in2;
  double * restrict res;

  const int N = 64;

  __m128d a[UNROLL];
  __m128d b[UNROLL];
  __m128d temp[UNROLL/2];

  res = C;
  in1 = A;
  in2 = B;
  /*
  
  //t correspond to i
  for(int t = 0;t <N;t++){
    in2 = B;//&B[(k)*N+j];

    //l correspond to k
    for(int l = 0; l<N;l+= 1 ){
      for(int tt =0; tt< UNROLL; tt++)
	a[tt] =  _mm_load1_pd(&in1[l+tt]);
      //a[0] =  _mm_load1_pd(&in1[l]);

      //m corresponds to j
      for(int m = 0; m<N;m+=2){
	for(int tt =0; tt< UNROLL;tt++)
	  b[tt] =  _mm_load_pd(&in2[m]);
	//b[tt] =  _mm_load_pd(&in2[(tt)*Stride_B+m]);
	//b[0] =  _mm_load_pd(&in2[m]);

	//set up
	for(int tt =0; tt< UNROLL/2;tt++)
	  temp[tt] =  _mm_add_pd(b[2*tt], b[2*tt+1]);

	__m128d c   = _mm_load_pd(&res[m]);
	b[0] =  _mm_mul_pd(a[0], b[0]);

	c =  _mm_add_pd(c, b[0] );

	_mm_store_pd(&res[m], c);
      }
      in2+=Stride_B;
    }

 res+=Stride_C;
 in1+=Stride_A;

  }


  */


  

  //t correspond to i
  for(int t = 0;t <N;t++){
    in2 = B;//&B[(k)*N+j];

    //l correspond to k
    for(int l = 0; l<N;l+= UNROLL ){
      //set up
      for(int tt =0; tt< UNROLL; tt++)
	a[tt] =  _mm_load1_pd(&in1[l+tt]);

      //m corresponds to j
      for(int m = 0; m<N;m+=2){
	//set up
	for(int tt =0; tt< UNROLL;tt++)
	  b[tt] =  _mm_load_pd(&in2[(tt)*Stride_B+m]);

	__m128d c   = _mm_load_pd(&res[m]);
	//set up
	for(int tt =0; tt< UNROLL;tt++)
	  b[tt] =  _mm_mul_pd(a[tt], b[tt]);
	//set up
	for(int tt =0; tt< UNROLL/2;tt++)
	  temp[tt] =  _mm_add_pd(b[2*tt], b[2*tt+1]);

	if(UNROLL == 1)
	  c =  _mm_add_pd(c, b[0] );
	else if(UNROLL == 2)
	  c =  _mm_add_pd(c, _mm_add_pd(b[0], b[1]) );
	else
	  for(int tt =0; tt< UNROLL/4;tt++)
	    c =  _mm_add_pd(c, _mm_add_pd(temp[2*tt], temp[2*tt+1]));

	_mm_store_pd(&res[m], c);
      }
	    in2+=Stride_B*UNROLL;
    }

	  res+=Stride_C;
	  in1+=Stride_A;

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
	    __m128d a =  _mm_load1_pd(in1+l);//_mm_load_sd(&in1[l]);
	    //a = _mm_unpacklo_pd(a,a);

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


