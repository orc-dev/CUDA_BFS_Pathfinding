/**
 * @file          master.cu
 * @brief         Conduct performance testing of the double-source BFS 
 *                implementation. Outputs results to the console and writes 
 *                the generated path as CSV files to the specified file.
 *
 * @author        Xin Cai
 * @email         xcai72@wisc.edu
 * @date          Dec. 10, 2023
 *
 * @course        ME759: High Performance Computing for Engineering Application
 * @instructor    Professor Dan Negrut
 * @assignment    Final Project
 *
 * @module        module load nvidia/cuda/11.8.0 
 * @compile       nvcc master.cu bfs.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o cudapath
 */
#include "bfs.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <ratio>
#include <string>
#include <limits>
#include <iomanip>
#include <cstdio>
#include <algorithm>


/**
 * @brief Reads a map from a file and populates a 1D char array
 *
 * @param filename The name of the file containing the map
 * @param imap     pointer to array
 */
void read_map(const std::string& filename, int* imap) {
    // create an input file stream
    std::ifstream map_file(filename);

    // check if the file is successfully opened
    if (!map_file.is_open()) {
        std::cout << "ERROR: fail to open \'" 
                  << filename << "\'" << std::endl;
        return;
    }

    // read file, update the least sigificant bit
    char ch;
    int i = 0;
    while ((ch = map_file.get()) != EOF) {
        if (ch == '0') {
            imap[i] = 0x7F;  // binary: 0111 1111
            i++;
        }
        else if (ch == '1') {
            imap[i] = 0;
            i++;
        }
    }
    map_file.close();
}


/**
 * @brief Checks if a cell in a 6-way map is within bounds given the 
 *        relative direction of the base cell.
 * 
 * @param r Row index
 * @param c Column index
 * @param dir Direction (TLF, TRT, BLF, BRT, LHS, RHS)
 * @return True if the neighbor cell is within bounds, false otherwise.
 */
bool in_bound(const int r, const int c, const int dir) {
    // constant parity-flags and last-column index
    const int  end  = (N - 1);
    const bool odd  = (r & 1) > 0;
    const bool even = !odd;
    
    switch(dir) {
        case TLF:
            return (odd) || (r > 0 && c > 0);
        
        case TRT:
            return (r > 0) && (even || c < end);

        case BLF:
            return (r < end) && (odd || r > 0);

        case BRT:
            return (even) || (r < end && c < end);

        case LHS:
            return (c > 0);

        case RHS:
            return (c < end);

        default:
            return false;
    }
}


/**
 * @brief Encode a character map to a 6-way map. For each non-border cell,
 *        it has six immediate neighbor cells.
 *
 *        row-0    (   ) (   ) (   )
 *        row-1       (TLF) (TRT) (   )
 *        row-2    (LHS) ( * ) (RHS)
 *        row-3       (BLF) (BRT) (   )
 * 
 * @param imap  Pointer to the input character map
 */
void encode_map(int* imap) {
    // for each row
    for (int bgn = 0, end = N-1, row = 0; row < N; bgn += N, end += N, ++row) {
        // update accessible bits of the first and last cell
        imap[bgn] &= ~LHS;
        imap[end] &= ~RHS;

        if ((row & 1) == 0)
            imap[bgn] &= ~(TLF | BLF);
        else
            imap[end] &= ~(TRT | BRT);
    }

    // update accessible bits of cells in the first and last row
    for (int bgn = 0, end = N * (N - 1); bgn < N; ++bgn, ++end) {
        imap[bgn] &= ~(TLF | TRT);
        imap[end] &= ~(BLF | BRT);
    }

    // static data for neighbor directions and offsets
    const int DIR[6] = { TLF, TRT, BLF, BRT, LHS, RHS };
    const int OPP[6] = { BRT, BLF, TRT, TLF, RHS, LHS };
    const int OFF[2][6] = {
        { -1-N,  -N, N-1,   N, -1, 1 },   // offsets for even rows
        {   -N, 1-N,   N, 1+N, -1, 1 }    // offsets for odd rows
    };

    // 1D index variable of the map
    int k = 0;

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c, ++k) {
            // skip non-obstacle cells
            if (imap[k] > 0) continue;

            // get the row parity
            const int parity = (r & 1);

            // update accessible bits of all reachable cells near this obstacle
            for (int i = 0; i < 6; ++i)
                if (in_bound(r, c, DIR[i]))
                    imap[k + OFF[parity][i]] &= ~OPP[i];
        }
    }
}


/**
 * @brief Initialize the output map
 *
 * @param imap  Pointer to the input map
 * @param sid   Source node ID
 * @param tid   Target node ID
 * @param omap  Pointer to the output map
 */
void init_omap(const int* imap, const int sid, const int tid, int* omap) {
    std::copy(imap, imap + NSQ + 2, omap);
    omap[sid] |= (DIRTY) | (1 << FOC_SHF);
    omap[tid] |= (DIRTY) | (1 << FOC_SHF) | (GROUP);
    omap[EP_0] = -1;
    omap[EP_1] = -1;
}


/**
 * @brief Writes a path to a CSV file.
 *
 * @param path      The path to be written to the CSV file
 * @param out_file  The base name of the output CSV file
 * @param id        An identifier used in the filename
 */
void write_path_to_csv(const std::vector<std::vector<int>>& path,
    const std::string& out_file, const int id) {
    
    std::string filename = out_file + "_" + std::to_string(id) + ".csv";
    // open the file for writing
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: fail to open \'" 
                  << filename << "\'" << std::endl;
        return;
    }
    // write data to the CSV file
    file << "x,y\n";
    for (const auto& point : path) {
        file << point[0] << "," << point[1] << "\n";
    }
    file.close();
}


/**
 * @brief Measure the runtime of a GPU function using CUDA events.
 *
 * @param omap Pointer to the output map.
 * @return Elapsed time in ms.
 */
float measure_gpu_runtime(int* omap) {
    // event creation
    float gpu_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // start, running and synchronize
    cudaEventRecord(start);
    {
        bfs(omap);
    }
    cudaDeviceSynchronize();
    
    // ending synchronize and compute elaspsed time
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_time, start, end);

    // event destroy
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return gpu_time;
}


/**
 * @brief   Rebuild the path from the target node to the source node 
 *          using the output map.
 *
 * @param   omap      Pointer to the output map.
 * @param   src_id    Source node ID.
 * @param   tgt_id    Target node ID.
 * @param   path      Vector of vectors representing the rebuilt path.
 */
void rebuild_path(const int* omap, const int src_id, const int tgt_id,
    std::vector<std::vector<int>>& path) {
    int k;
    // build the source subpath
    k = omap[EP_0];
    path.push_back({k / N, k % N});

    while (k != src_id) {
        k = omap[k] >> DAT_SHF;
        path.push_back({k / N, k % N});
    }
    // revserse path letting the source node in front
    std::reverse(path.begin(), path.end());

    // build the target subpath
    k = omap[EP_1];
    path.push_back({k / N, k % N});

    while (k != tgt_id) {
        k = omap[k] >> DAT_SHF;
        path.push_back({k / N, k % N});
    }
}


/**
 * @brief   Measure the runtime of the path rebuilding process.
 *
 * @param   omap    Pointer to the output map.
 * @param   src_id  Source node ID.
 * @param   tgt_id  Target node ID.
 * @param   path    Vector of vectors representing the rebuilt path.
 *
 * @return  Elapsed time in milliseconds.
 */
double timing_path_rebuild(const int* omap, const int src_id, const int tgt_id,
    std::vector<std::vector<int>>& path) {

    // prepare timing tools
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_ms;

    // measure the runtime
    start = std::chrono::high_resolution_clock::now();
    { 
        rebuild_path(omap, src_id, tgt_id, path);
    }
    end = std::chrono::high_resolution_clock::now();

    // compute duration
    using dur_in_ms = std::chrono::duration<double, std::milli>;
    duration_ms = std::chrono::duration_cast<dur_in_ms>(end - start);

    return duration_ms.count();
}


/**
 * @brief Runs pathfinding tests and outputs results to CSV files.
 *
 * @param map_file  The file containing the map for pathfinding
 * @param out_file  The base name for output CSV files
 * @param src_ids   Vector of source points for pathfinding tasks
 * @param tgt_ids   Vector of target points for pathfinding tasks
 */
void run_test(const std::string map_file, const std::string out_file,
    const std::vector<int>& src_ids, const std::vector<int>& tgt_ids) {
    // display settings
    const std::string separator(40, '-');
    printf("%s\n", separator.c_str());
    printf("map file:   %s\n", map_file.c_str());
    
    // allocation with cuda managed memory
    int *imap;
    int *omap;
    const int buf_size = NSQ + 2;
    cudaMallocManaged(&imap, buf_size * sizeof(int));
    cudaMallocManaged(&omap, buf_size * sizeof(int));

    // read and encode map
    read_map(map_file, imap);
    encode_map(imap);
    
    // warm-up run
    init_omap(imap, src_ids[0], tgt_ids[0], omap);
    measure_gpu_runtime(omap);

    float  gpu_time;
    double rbd_time;

    // display table header
    std::string header_fmt = "%-4s%-11s%-15s%-15s\n";
    std::string record_fmt = "%-4zu%-11.4f%-15.4lf%-15zu\n";
    printf(header_fmt.c_str(), "No.", "gpu_time", "rebuild_time", "path_size");

    for (size_t i = 0; i < src_ids.size(); ++i) {
        // timing cuda pathfinding
        init_omap(imap, src_ids[i], tgt_ids[i], omap);
        gpu_time = measure_gpu_runtime(omap);

        // timing cpu path rebuild
        std::vector<std::vector<int>> path;
        rbd_time = timing_path_rebuild(omap, src_ids[i], tgt_ids[i], path);
        
        // display results
        printf(record_fmt.c_str(), i, gpu_time, rbd_time, path.size() - 1);
        write_path_to_csv(path, out_file, i);
    }
    // deallocation
    cudaFree(imap);
    cudaFree(omap);
}


/**
 * Organize data and run tests on the sparse map and the dense map.
*/
int main(int argc, char *argv[]) {
    // pathfinding on sparse map
    {
        const std::string map_file = "../maps/map_512_sparse.txt";
        const std::string out_file = "../out/path_S6_DW_plus";

        std::vector<int> src_ids = {
            0 * N + 0,
            511 * N + 0,
            270 * N + 150,
            270 * N + 150,
            125 * N + 500,
        };

        std::vector<int> tgt_ids = {
            511 * N + 511,
            511 * N + 511,
            231 * N + 375,
            232 * N + 375,
            340 * N + 485,
        };

        run_test(map_file, out_file, src_ids, tgt_ids);
    }

    //pathfinding on dense map
    {
        const std::string map_file = "../maps/map_512_dense.txt";
        const std::string out_file = "../out/path_D6_DW_plus";
        
        std::vector<int>src_ids = {
            1 * N + 4,
            434 * N + 1,
            62 * N + 100,
            62 * N + 346,
            246 * N + 25,
        };

        std::vector<int>tgt_ids = {
            509 * N + 509,
            194 * N + 458,
            351 * N + 434,
            482 * N + 182,
            223 * N + 365,
        };

        run_test(map_file, out_file, src_ids, tgt_ids);
    }
    return 0;
}