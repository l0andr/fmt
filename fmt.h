#ifndef FMT_H_INCLUDED
#define FMT_H_INCLUDED
// Transpose  matrix A and store in B. Simple version.
// n , m : sizes of matrix. B should be pre-allocated and have the same size as A.
void transpose_simple(float *A, float *B, const size_t n, const size_t m);

// Transpose  matrix A and store in B. Simple + multi-threading version.
// n , m : sizes of matrix. B should be pre-allocated and have the same size as A.
void transpose_simple_parallel(float *A, float *B, const size_t n, const size_t m, size_t nthreads);

// Transpose  matrix A and store in B. Using SSE4 instruction
// n , m : sizes of matrix. B should be pre-allocated and have the same size as A.
void transpose_SSE(float *A, float *B, const size_t n, const size_t m);

// Inplace transpose of square matrix.
// The number of elements of the matrix must be a multiple of 4.
void transpose_SSE_inplace(float *A, const size_t n);

// Transpose  matrix A and store in B (Alghoritm with sub-blocks)
// n , m : sizes of matrix. B should be pre-allocated and have the same size as A.
// block_size - size of submatrix to transpose, actually ahould be smaller or eq of Processor cash size
void transpose_blocks(float *A, float *B, const size_t n, const size_t m,const size_t block_size);

// Transpose  matrix A and store in B. (Alghoritm with sub-blocks and multi-threding)
// n , m : sizes of matrix. B should be pre-allocated and have the same size as A.
// block_size - size of submatrix to transpose, actually should be smaller or eq. to processor cash size
// nthreads - number of computational threads (normally when nthreads = Number of CPU cores )
void transpose_blocks_parallel(float *A, float *B, const size_t n, const size_t m,const size_t block_size,const size_t nthreads);
#endif // FMT_H_INCLUDED
