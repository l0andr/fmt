# Fast Matrix Transposition 


This repository contains functions with different approaches to Transposition of Matrix and performance tests of them.

We sequentially implement : <br>
1. Naive matrix transposition (single-thread)<br> 
2. Parallel naive matrix transposition (multi-threads)<br> 
3. SSE matrix transposition (single-thread) <br>
4. SSE Block matrix  transposition (single-thread) <br>  
5. and fastest variant <b>SSE Parallel Block matrix transposition</b> (multi-threads) <br> 

## Results of performance tests:

<img src="img/perf_test_3.png" align=center> <br>

All tests were performed on <i> "Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz" with 16 GB of DDR4 RAM </i> <br>
As can be seen significant difference apears starting with matrix about 2000x2000 <br> 

<b> SSE Block matrix transposition </b> approach faster than any other single-thread approaches <br>
<b> SSE Parallel Block matrix transposition </b> fastest at all <br>   


## How-to-start tests:

### To start performance test build and run following: 

FastMatrixTransposition [matrix_size] [block_size] [number_of_threads] [number_of_tests performance tests for each approach] <br>

Will outputed average times for each approah<br>   

### Example of output:  <br>
8000,194659.156250,97701.046875,140561.281250,86731.703125,62631.093750 <br>

### Format of output: <br>
[matrix_size],[naive approach], [parallel naive approach], [SSE matrix transposition], [Block transposition], [Block SSE parallel transposition]<br>

All times in nano seconds<br>
