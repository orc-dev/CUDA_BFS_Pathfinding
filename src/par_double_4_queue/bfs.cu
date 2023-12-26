/**
 * @file          bfs.cu
 * @brief         Implementation of a CUDA kernel for double-source BFS.
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
 * @brief CUDA kernel function for exploring level of cells in the BFS.
 * 
 * @param emap Pointer to the encoded map
 * @param omap Pointer to the output map
 * @param curr_q Pointer to the current queue
 * @param next_q Pointer to the next queue
 * @param offset Offset for current cell to the neighbor
 * @param direction Current exploring direction
 */
__global__ 
void explore(const std::uint8_t* emap, int* omap, int* curr_q, 
             int* next_q, int offset, int direction) {
    // variables initialization
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_size  = curr_q[QID];
    
    // thread terminal: exceeds task size, neib cell is inaccessible, 
    if (idx >= task_size) return;
    
    const int in_path_t  = curr_q[idx] < 0;
    const int cid        = in_path_t ? ~curr_q[idx] : curr_q[idx];
    const std::uint8_t k = emap[cid];
    int nid;
    int enqueue_id;

    // current cell is obstacle
    if (k & CLOSED) return;
    
    nid = cid + offset;
    // termination: two subpath reach to each other
    if ((k & direction) == 0 && !in_path_t && omap[nid] < 0) {
        omap[SID] = cid; 
        omap[TID] = ~nid;
        return;
    }

    // if find a new cell in frontier of next level, enqueue it
    if ((k & direction) == 0 && omap[nid] == DEF_VAL) {    
        omap[nid] = in_path_t ? ~cid : cid;

        // atomic add to get and increment a global index of the queue
        enqueue_id = atomicAdd(&next_q[QID], 1);
        next_q[enqueue_id] = (in_path_t) ? ~nid : nid;
    }
}


/**
 * @brief Perform breadth-first search (BFS) on a grid using CUDA,
 *        two queues (raw) are in used to support the BFS.
 * 
 * @param emap Pointer to the encoded map
 * @param omap Pointer to the output map
 * @param curr_q Pointer to the current queue
 * @param next_q Pointer to the next queue
 */
__host__
void bfs(const std::uint8_t* emap, int* omap, int** curr_q, int** next_q) {
    // static data for neighbor directions and offsets
    const int dir[4] = { TOP, BOT, LHS, RHS };
    const int off[4] = {  -N,   N,  -1,   1 }; 

    // static and runtime variables for kernel configurations
    const int num_thd = 128;
    int num_blk;
    int queue_rtsize;
    int **temp_q;
    
    while (omap[SID] == DEF_VAL) {
        // kernel configuration
        queue_rtsize = (*curr_q)[QID];
        num_blk = (queue_rtsize + num_thd - 1) / num_thd;
        
        // invoke kernel functions
        for (int i = 0; i < 4; ++i) {
            explore<<<num_blk, num_thd>>>(
                emap, omap, *curr_q, *next_q, off[i], dir[i]
            );
            cudaDeviceSynchronize();
        }
        // swap and reset buffer
        (*curr_q)[QID] = 0;
        temp_q = curr_q;
        curr_q = next_q;
        next_q = temp_q;
    }
}
