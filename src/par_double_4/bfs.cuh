/**
 * @file          bfs.cuh
 * @brief         Defines the function prototype `bfs()` and related macros to 
 *                implement a parallel version of double-source BFS in CUDA.
 *
 * @author        Xin Cai
 * @email         xcai72@wisc.edu
 * @date          Nov. 24, 2023
 *
 * @course        ME759: High Performance Computing for Engineering Application
 * @instructor    Professor Dan Negrut
 * @assignment    Final Project   
 */
#ifndef BFS_CUH
#define BFS_CUH

#include <cstdint>

// Macros for constants
#define DEF_VAL 1 << 30
#define N       512
#define NSQ     N * N
#define SID     NSQ * 2
#define TID     SID + 1

// Macros of bitmasks
#define TOP     16
#define BOT     8
#define LHS     4
#define RHS     2
#define OPEN    1

__host__ void bfs(const std::uint8_t* emap, int* omap);

#endif