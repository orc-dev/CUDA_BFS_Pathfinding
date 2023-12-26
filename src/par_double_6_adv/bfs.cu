/**
 * @file          bfs.cu
 * @brief         Implementation of a CUDA kernel for double-source BFS for
 *                6-way map.
 *
 * @author        Xin Cai
 * @email         xcai72@wisc.edu
 * @date          Dec. 10, 2023
 *
 * @course        ME759: High Performance Computing for Engineering Application
 * @instructor    Professor Dan Negrut
 * @assignment    Final Project   
 */
#include "bfs.cuh"
#include <unordered_map>
#include <vector>

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
    const int DIR[6] = { TLF, TRT, BLF, BRT, LHS, RHS };
    const int OPP[6] = { BRT, BLF, TRT, TLF, RHS, LHS };
    const int OFF[2][6] = {
        { -1-N,  -N, N-1,   N, -1, 1 },   // offsets for even rows
        {   -N, 1-N,   N, 1+N, -1, 1 }    // offsets for odd rows
    };
    const int par_key = (idx / N) & 1;    // row parity as key

    // organize bitmap messages for 'clean' neighbors
    const int curr_grp = (curr & GROUP);
    const int next_dat = (idx << DAT_SHF) | (curr_grp) | (DIRTY) | (next_foc);
    
    // check six neighbors
    int nid, neib, off_p;
    
    for (int i = 0; i < 6; ++i) {
        // check next if the current neib is inaccessible
        if (!(curr & DIR[i])) 
            continue;
        
        // access a neighbor
        nid = idx + OFF[par_key][i];
        neib = omap[nid];

        // write next-level data to a 'clean' neighbor cell
        if (!(neib & DIRTY)) {
            omap[nid] = (next_dat) | (neib & ~OPP[i]);
        } 
        // encapsulate result to the entry point if hitting the other set
        else if ((curr_grp) != (neib & GROUP)) {
            off_p = ((neib & focus) == 0);
            omap[EP_0 + off_p] = (next_dat) | (DIR[i]);
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
    
    // create a map from direction to offset
    std::unordered_map<int, std::vector<int>> dir_offset = {
        { TLF, { -1-N,  -N } },
        { TRT, {   -N, 1-N } },
        { BLF, { -1+N,   N } },
        { BRT, {   N,  1+N } },
        { LHS, {  -1,   -1 } },
        { RHS, {   1,    1 } }
    };

    // decapsulate message at the entry point
    const int offset = omap[EP_0] < 0;
    const int bitmap = omap[EP_0 + offset];

    // extract index of current and neighbor cells
    const int cid     = (bitmap >> DAT_SHF);
    const int dir_key = (bitmap & 0x7f);   // direction key
    const int par_key = (cid / N) & 1;     // parity key
    const int nid     = dir_offset[dir_key][par_key] + cid;

    // compute offsets for entry points
    const int off_c = (bitmap & GROUP) >> 12;
    const int off_n = (off_c ^ 1);

    // write results to entry positions
    omap[EP_0 + off_c] = cid;
    omap[EP_0 + off_n] = nid;
}
