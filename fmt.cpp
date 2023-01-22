
#ifndef FMT_CPP_INCLUDED
#define FMT_CPP_INCLUDED

#include <xmmintrin.h>
#include <iostream>
#include <cstring>
#include <cassert>
#include <thread>
inline void transpose4x4_SSE_all(float *A, float *B, const size_t lda, const size_t ldb) {
    __m128 row1 = _mm_loadu_ps(&A[0*lda]);
    __m128 row2 = _mm_loadu_ps(&A[1*lda]);
    __m128 row3 = _mm_loadu_ps(&A[2*lda]);
    __m128 row4 = _mm_loadu_ps(&A[3*lda]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_storeu_ps(&B[0*ldb], row1);
     _mm_storeu_ps(&B[1*ldb], row2);
     _mm_storeu_ps(&B[2*ldb], row3);
     _mm_storeu_ps(&B[3*ldb], row4);
}
inline void transpose4x4_SSE_aligned(float *A, float *B, const size_t lda, const size_t ldb) {
    __m128 row1 = _mm_load_ps(&A[0*lda]);
    __m128 row2 = _mm_load_ps(&A[1*lda]);
    __m128 row3 = _mm_load_ps(&A[2*lda]);
    __m128 row4 = _mm_load_ps(&A[3*lda]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_store_ps(&B[0*ldb], row1);
     _mm_store_ps(&B[1*ldb], row2);
     _mm_store_ps(&B[2*ldb], row3);
     _mm_store_ps(&B[3*ldb], row4);
}
void thread_func_transpose_simple(float *A, float *B, const size_t n,const size_t m,size_t is, size_t ie)
{
    for (size_t i=is;i<ie;i++)
     for (size_t j=i;j<m;j++)
        B[j*n+i] = A[i*m+j];
}

void transpose_simple(float *A, float *B, const size_t n, const size_t m)
{
  for (size_t i=0;i<n;i++)
     for (size_t j=i;j<m;j++)
        B[j*n+i] = A[i*m+j];
}
void transpose_simple_parallel(float *A, float *B, const size_t n, const size_t m, size_t nthreads)
{

    std::thread *comp_threads=new std::thread[nthreads];
    for (size_t t=0;t<nthreads;t++)
        {
             size_t is,ie;
             is=t*(n/nthreads);

            if (t<(nthreads-1))
             ie=(t+1)*(n/nthreads);
            else
             ie=n;
             comp_threads[t]=std::thread(thread_func_transpose_simple,A,B,n,m,is,ie);
        }
    for (size_t t=0;t<nthreads;t++) comp_threads[t].join();
    delete [] comp_threads;
}


void transpose_SSE_all(float *A, float *B, const size_t n, const size_t m) {

    size_t nf=(n/4)*4;
    size_t mf=(m/4)*4;

    for(int i=0; i<nf; i+=4) {
        for(int j=0; j<mf; j+=4) {
            transpose4x4_SSE_all(&A[i*m +j], &B[j*n + i], m, n);
        }
        for(int k=i;k<i+4;k++)
        for(int j=mf; j<m; j+=1) B[j*n + k]=A[k*m +j];
    }

    for(int i=nf; i<n; i+=1) {

        for(int j=0; j<m; j+=1) {

            B[j*n + i]=A[i*m +j];

        }
    }
}
//The number of elements of the matrix must be a multiple of 4.
void transpose_SSE_aligned(float *A, float *B, const size_t n, const size_t m) {

    size_t nf=(n/4)*4;
    size_t mf=(m/4)*4;
     assert(nf == n && mf==m); //The number of elements of the matrix must be a multiple of 4. Actually. Always.

    for(int i=0; i<nf; i+=4) {
        for(int j=0; j<mf; j+=4) {
            transpose4x4_SSE_aligned(&A[i*m +j], &B[j*n + i], n, m);
        }
    }

}
void transpose_SSE(float *A, float *B, const size_t n, const size_t m) {

    size_t nf=(n/4)*4;
    size_t mf=(m/4)*4;
    if (nf==n && mf==m)
        transpose_SSE_aligned(A, B, n, m);
       else
        transpose_SSE_all(A, B, n, m);

}



// Inplace transpose of square matrix.
// The number of elements of the matrix must be a multiple of 4.

void transpose_SSE_inplace(float *A, const size_t n) {


    size_t nf=(n/4)*4;

    assert(nf == n); //The number of elements of the matrix must be a multiple of 4. Actually. Always.

    float *tmp_array=new float[16];
    for(int i=0; i<n; i+=4) {
     int j=i;
     transpose4x4_SSE_aligned(&A[i*n +j], &A[j*n + i], n, n);
    }
    for(int i=0; i<n; i+=4) {
        for(int j=i+4; j<n; j+=4) {

            for (int k=0;k<4;k++) memcpy(&tmp_array[k*4],&((&A[j*n+i])[k*n]),sizeof(float)*4);
               transpose4x4_SSE_aligned(&A[i*n +j], &A[j*n + i], n, n);
              transpose4x4_SSE_aligned(&tmp_array[0], &A[i*n + j], 4, n);
        }
    }
    delete [] tmp_array;

}
void transpose_blocks(float *A, float *B, const size_t n, const size_t m,const size_t block_size)
{

    float *tmp_array;
    tmp_array=new float[block_size*block_size];
     size_t nf=(n/block_size)*block_size;
    size_t mf=(m/block_size)*block_size;
    for (size_t i=0;i<nf;i+=block_size)
        for (size_t j=0;j<mf;j+=block_size)
        {
           for (size_t k=0;k<block_size;k++) memcpy(&tmp_array[k*block_size],&A[i*m+j+k*m],sizeof(float)*block_size);
           transpose_SSE_inplace(tmp_array, block_size);
           for (size_t k=0;k<block_size;k++) memcpy(&B[j*n+i+k*n],&tmp_array[k*block_size],sizeof(float)*block_size);
        }
    delete [] tmp_array;
    if (n != nf)
    {   size_t bs=n-nf;
        tmp_array=new float[bs*m];
        transpose_SSE_all(&A[nf*m],tmp_array, bs,m);
        for (size_t k=0;k<m;k++) memcpy(&B[nf+k*n],&tmp_array[k*bs],sizeof(float)*bs);
        delete [] tmp_array;
    }
    if (m != mf)
    {   size_t bs=m-mf;
        tmp_array=new float[bs*n];
        float *tmp_array2=new float[bs*n];
        for (size_t k=0;k<n;k++) memcpy(&tmp_array[k*bs],&A[k*m+mf],sizeof(float)*bs);
        transpose_SSE_all(tmp_array,&B[mf*n], n,bs);
        delete [] tmp_array;
    }
}
void thread_func_transpose(float *A, float *B,float *tmp_array,const size_t n, const size_t m,const size_t block_size,const size_t is, const size_t ie)
{
   // std::cout<<"Thread start:"<<is<<" "<<ie<<std::endl;
   tmp_array=new float[block_size*block_size];
    size_t mf=(m/block_size)*block_size;
    for (size_t i=is;i<ie;i+=block_size)
        for (size_t j=0;j<mf;j+=block_size)
        {
           for (size_t k=0;k<block_size;k++) memcpy(&tmp_array[k*block_size],&A[i*m+j+k*m],sizeof(float)*block_size);
           transpose_SSE_inplace(tmp_array, block_size);
           for (size_t k=0;k<block_size;k++) memcpy(&B[j*n+i+k*n],&tmp_array[k*block_size],sizeof(float)*block_size);
        }
   delete [] tmp_array;
   // std::cout<<"Thread end:"<<is<<" "<<ie<<std::endl;
}
void transpose_blocks_parallel(float *A, float *B, const size_t n, const size_t m,const size_t block_size,size_t nthreads)
{
    float **tmp_array_pool;
    size_t nf=(n/block_size)*block_size;
    size_t num_t_blocks=(nf/block_size)/nthreads;
    if (num_t_blocks==0)
     {
        nthreads=(nf/block_size);
        num_t_blocks=1;
     }
    tmp_array_pool=new float*[nthreads];
    std::thread *comp_threads=new std::thread[nthreads];


    for (size_t t=0;t<nthreads;t++)
        {
            // tmp_array_pool[t]=new float[block_size*block_size];
             size_t is,ie;
             is=t*num_t_blocks*block_size;

            if (t<(nthreads-1))
             ie=(t+1)*num_t_blocks*block_size;
            else
             ie=nf;

             comp_threads[t]=std::thread(thread_func_transpose,A,B,tmp_array_pool[t],n,m,block_size,is,ie);

        }
    for (size_t t=0;t<nthreads;t++) comp_threads[t].join();
    delete [] comp_threads;
    delete [] tmp_array_pool;
    size_t mf=(m/block_size)*block_size;


    if (n != nf)
    {   size_t bs=n-nf;
        float *tmp_array=new float[bs*m];
        transpose_SSE_all(&A[nf*m],tmp_array, bs,m);
        for (size_t k=0;k<m;k++) memcpy(&B[nf+k*n],&tmp_array[k*bs],sizeof(float)*bs);
        delete [] tmp_array;
    }
    if (m != mf)
    {   size_t bs=m-mf;
        float *tmp_array=new float[bs*n];
        float *tmp_array2=new float[bs*n];
        for (size_t k=0;k<n;k++) memcpy(&tmp_array[k*bs],&A[k*m+mf],sizeof(float)*bs);
        transpose_SSE_all(tmp_array,&B[mf*n], n,bs);

        delete [] tmp_array;
    }



}


#endif // FMT_CPP_INCLUDED
