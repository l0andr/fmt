# Fast Matrix Transposition 


This repository contains functions with different approaches to Transposition of Matrix and performance tests of them.

We sequentially implement : <br>
1. Naive matrix transposition <br> 
2. Parallel naive matrix transposition <br> 
3. SSE matrix transposition <br>
4. Block SSE matrix  transposition <br>  
5. and fastest variant <b>Parallel Block SSE matrix transposition</b> <br> 

## How-to-start tests:

### To start performance test build and run following: 

FastMatrixTransposition [matrix_size] [block_size] [number_of_threads] [number_of_tests performance tests for each approach] <br>

Will outputed average times for each approah<br>   

### Example of output:  <br>
8000,194659.156250,97701.046875,140561.281250,86731.703125,62631.093750 <br>

### Format of output: <br>
[matrix_size],[naive approach], [parallel naive approach], [SSE matrix transposition], [Block transposition], [Block SSE parallel transposition]<br>

All times in nano seconds<br>
