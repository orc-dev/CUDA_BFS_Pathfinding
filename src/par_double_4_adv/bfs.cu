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
 * @brief Perform the next step of Breadth-First Search
 *        algorithm for parallel traversal.
 *
 * @param omap  Pointer to the output map.
 * @param focus Current focus bits for traversal.
 */
__global__ void bfs_next_step(int* omap, const int focus) {
    // get the current index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NSQ) return;

    const int next_foc = (~focus & FOCUS);
    const int curr = omap[idx];
    
    // clear focus bits for a cell that was focused in the last iteration
    if (curr & next_foc) {
        omap[idx] &= ~FOCUS;
        return;
    }

    // return if this is not currently focusing cell
    if (!(curr & focus))
        return;

    // static data for neighbor directions and offsets
    const int dir[4] = { TOP, BOT, LHS, RHS };
    const int opp[4] = { BOT, TOP, RHS, LHS };
    const int off[4] = {  -N,   N,  -1,   1 };

    const int curr_grp = (curr & GROUP);
    const int next_dat = (idx << DAT_SHF) | (curr_grp) | (DIRTY) | (next_foc);
    
    int nid;
    int neib;
    int offset;

    // check four neighbors
    for (int i = 0; i < 4; ++i) {
        // check next if the current neib is inaccessible
        if (!(curr & dir[i])) 
            continue;
        
        nid = idx + off[i];
        neib = omap[nid];

        // write next-level data to clean neighbor cell
        if (!(neib & DIRTY)) {
            omap[nid] = (next_dat) | (neib & ~opp[i]);
        } 
        // encapsulate result to the entry point if hitting the other set
        else if ((curr_grp) != (neib & GROUP)) {
            offset = ((neib & focus) == 0);
            omap[EP_0 + offset] = (next_dat) | (dir[i]);
            return;
        }
    }
}


/**
 * @brief Host function for breadth-first search (BFS) on a graph.
 *
 * @param omap Pointer to the output map.
 */
__host__ void bfs(int* omap) {
    // constants for kernel configurations
    const int thd_num = 128;
    const int blk_num = (NSQ + thd_num - 1) / thd_num;
    int focus = (1 << FOC_SHF);

    while (omap[EP_0] < 0 && omap[EP_1] < 0) {
        // invoke kernel to do the pathfinding
        bfs_next_step<<<blk_num, thd_num>>>(omap, focus);
        cudaDeviceSynchronize();

        // switch between two focus status bit
        focus = ~focus & FOCUS;
    }

    // decapsulate message at the entry point
    const int offset = omap[EP_0] < 0;
    const int bitmap = omap[EP_0 + offset];

    // extract index of current and neighbor cells
    const int cid = (bitmap >> DAT_SHF);
    const int nid = (bitmap & TOP) ? cid - N :
                    (bitmap & BOT) ? cid + N :
                    (bitmap & LHS) ? cid - 1 :
                                     cid + 1;

    // compute offsets for entry points
    const int off_c = (bitmap & GROUP) >> 12;
    const int off_n = (off_c ^ 1);

    omap[EP_0 + off_c] = cid;
    omap[EP_0 + off_n] = nid;
}
