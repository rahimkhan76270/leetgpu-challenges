#include <cuda_runtime.h>

// All pointers are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, 
                     int start_row, int start_col, int end_row, int end_col) {
    // TODO: Implement BFS shortest path using CUDA
    // 
    // grid: flattened 2D array of size rows*cols (0=free, 1=obstacle)
    // result: single element array to store the shortest path length (-1 if no path)
    // 
    // Grid indexing: grid[row * cols + col]
    // Movement directions: up (-1,0), down (1,0), left (0,-1), right (0,1)
    
}