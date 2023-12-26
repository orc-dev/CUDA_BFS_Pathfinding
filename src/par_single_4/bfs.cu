/**
 * @file          bfs.cu
 * @brief         Implement the single-source CUDA version of the BFS 
 *                pathfinding in a parallel fashion.
 *
 * @author        Xin Cai
 * @email         xcai72@wisc.edu
 * @date          Nov. 24, 2023
 *
 * @course        ME759: High Performance Computing for Engineering Application
 * @instructor    Professor Dan Negrut
 * @assignment    Final Project   
 */
#include "bfs.cuh"


/**
 * @brief   CUDA kernel to copy values from a temporary buffer to a main buffer.
 *
 * @param   omap  Pointer to the main buffer.
 */
__global__ void kernel_cpy(int* omap) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // copy from temp buffer to main buffer
    if (omap[idx] == DEF_VAL && omap[idx + NSQ] < DEF_VAL)
        omap[idx] = omap[idx + NSQ];
}


/**
 * @brief   CUDA kernel for breadth-first search (BFS) exploration on a graph.
 *
 * @param   emap  Pointer to the encoded map.
 * @param   omap  Pointer to the output map.
 */
__global__ void kernel_bfs(const std::uint8_t* emap, int* omap) {
    // constants and indices setup
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NSQ) return;

    const std::uint8_t k = emap[idx];

    // current cell is block or has been processed
    if ((k & OPEN) || omap[idx] < DEF_VAL)
        return;
    
    // static data for neighbor directions and offsets
    const int dir[4] = { TOP, BOT, LHS, RHS };
    const int off[4] = {  -N,   N,  -1,   1 }; 
    int nid;

    // check four neighbors
    for (int i = 0; i < 4; ++i) {
        nid = idx + off[i];
        // if neighbor is a frontier cell, write to buffer of current cell
        if ((k & dir[i]) == 0 && omap[nid] < DEF_VAL) {
            omap[idx + NSQ] = nid;
            return;
        }
    }
}


/**
 * @brief   Host function for breadth-first search (BFS) on a graph.
 *
 * @param   emap   Pointer to the encoded map.
 * @param   omap   Pointer to the output map.
 * @param   tgt_id Target node ID for BFS.
 */
__host__ void bfs(const std::uint8_t* emap, int* omap, const int tgt_id) {
    // constants for kernel configurations
    const int thd_num = 128;
    const int blk_num = (NSQ + thd_num - 1) / thd_num;
    
    // double-buffer and double kernel to do block-level synchronization
    while (omap[tgt_id] == DEF_VAL) {
        kernel_bfs<<<blk_num, thd_num>>>(emap, omap);
        cudaDeviceSynchronize();

        kernel_cpy<<<blk_num, thd_num>>>(omap);
        cudaDeviceSynchronize();
    }
}
