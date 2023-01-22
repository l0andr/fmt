#include <iostream>
#include "fmt.h"
#include <chrono>
#include <string>
using namespace std;

float *CreateTestMat(size_t n,size_t m)
{
    float *Mat;
    Mat=new float[n*m];
    for (size_t i=0; i<n; i++)
        for (size_t j=0; j<m; j++)
        {
            Mat[i*m+j]=(i+1)*100+(j+1);
        }
    return Mat;

}
void TransposePerfDisplay(size_t n_tests, size_t Msize,size_t block_size, size_t nthreads)
{

    size_t n=Msize,m=Msize;

    float aver_sup=0,aver_sup1=0,aver_sup2=0,aver_sup3=0.0,aver_sup4=0.0;
    std::chrono::time_point<std::chrono::steady_clock> start_time,end_time;
    std::chrono::nanoseconds elapsed_ns;
    for (size_t nn=0; nn<n_tests; nn++)
    {
        float *A = CreateTestMat(n,m);
        float *B = CreateTestMat(n,m);
        start_time = std::chrono::steady_clock::now();
        transpose_simple(A,B,n,m);
        end_time = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        aver_sup+=(float)elapsed_ns.count()/1000;
        start_time = std::chrono::steady_clock::now();
        transpose_simple_parallel(A,B,n,m,nthreads);
        end_time = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        aver_sup1+=(float)elapsed_ns.count()/1000;
        start_time = std::chrono::steady_clock::now();
        transpose_SSE(A,B,n,m);
        end_time = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        aver_sup2+=(float)elapsed_ns.count()/1000;
        start_time = std::chrono::steady_clock::now();
        transpose_blocks(A,B,n,m,block_size);
        end_time = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        aver_sup3+=(float)elapsed_ns.count()/1000;
        start_time = std::chrono::steady_clock::now();
        transpose_blocks_parallel(A,B,n,m,block_size,nthreads);
        end_time = std::chrono::steady_clock::now();
        elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        aver_sup4+=(float)elapsed_ns.count()/1000;
        delete [] A;
        delete [] B;

    }
    std::cout<<fixed <<Msize<<","<<aver_sup/n_tests<<","<<aver_sup1/n_tests<<","<<aver_sup2/n_tests<<","<<aver_sup3/n_tests<<","<<aver_sup4/n_tests<<std::endl;

}
int main(int argc, char* argv[])
{
    size_t nt=10;
    size_t ms=8000;
    size_t bs=100;
    size_t nthr=4;
    if (argc>1)
        ms=std::stoi(argv[1]);
    if (argc>2)
        bs=std::stoi(argv[2]);
    if (argc>3)
        nthr=std::stoi(argv[3]);
    if (argc>4)
        nt=std::stoi(argv[4]);
    if (bs<=0 || ms <= 00 || nthr <=0 || nt <= 0)
        std::cout<<" Traspose Perf. Test. Incorrect param"<<std::endl;
    if (bs>ms)
    {
        std::cout<<" Traspose Perf. Test. Incorrect param"<<std::endl;
        std::cout<<" block size should be lower than matrix size"<<std::endl;
    }
    if (nthr>ms)
    {
        std::cout<<" Traspose Perf. Test. Incorrect param"<<std::endl;
        std::cout<<" Number of threads should be lower than matrix size"<<std::endl;
    }
    TransposePerfDisplay(nt,ms,bs,nthr);

    return 0;
}
