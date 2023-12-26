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
#define N       512
#define NSQ     512 * 512
#define EP_0    NSQ          // entry point 0
#define EP_1    NSQ + 1      // entry point 1

// Macros of bitmasks
#define TOP     16
#define BOT     8
#define LHS     4
#define RHS     2
#define OPEN    1

#define DAT_SHF 13
#define FOC_SHF 9
#define GROUP   1 << 12
#define DIRTY   1 << 11
#define FOCUS   3 << 9

__host__ void bfs(int* omap);

#endif