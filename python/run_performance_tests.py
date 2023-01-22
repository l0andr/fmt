import matplotlib
import matplotlib.pyplot as plt
import subprocess
if __name__ == '__main__':
    block_size = 100
    n_threads = 8
    n_tests = 30
    matrix_sizes = []
    n_ap = []
    pn_ap = []
    sse_ap = []
    block_ap = []
    parallel_sse_block_ap = []
    numpy_ap = []
    for i in range(1,16):
        matrix_size = i*500
        out = subprocess.run(["../bin/Release/FastMatrixTransposition", f"{matrix_size} {block_size} {n_threads} {n_tests} "], stdout=subprocess.PIPE)
        res = out.stdout.decode().strip().split(',')
        print(res)
        matrix_sizes.append(float(res[0]))
        n_ap.append(float(res[1]))
        pn_ap.append(float(res[2]))
        sse_ap.append(float(res[3]))
        block_ap.append(float(res[4]))
        parallel_sse_block_ap.append(float(res[5]))
    plt.xlabel("Matrix size")
    plt.ylabel("Time in nanoseconds")
    plt.title(f"Performance of different methods of matrix transposition\n Block size:{block_size} Number of threads:{n_threads} ")
    plt.plot(matrix_sizes, n_ap, label='Naive transposition')
    plt.plot(matrix_sizes, pn_ap, label='Parallel naive transposition')
    plt.plot(matrix_sizes, sse_ap, label='SSE transposition')
    plt.plot(matrix_sizes, block_ap, label='SSE block transposition')
    plt.plot(matrix_sizes, parallel_sse_block_ap, label='SSE parallel block transposition')

    plt.legend()
    plt.grid()
    plt.show()
