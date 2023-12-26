/**
 * @file          master.cu
 * @brief         Conduct performance testing of the double-source BFS 
 *                implementation. Outputs results to the console and writes 
 *                the generated path as CSV files to the specified file.
 *
 * @author        Xin Cai
 * @email         xcai72@wisc.edu
 * @date          Nov. 24, 2023
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
void read_map(const std::string& filename, char* imap) {
    std::ifstream map_file(filename);

    // check if the file is successfully opened
    if (!map_file.is_open()) {
        std::cout << "ERROR: fail to open \'" 
                  << filename << "\'" << std::endl;
        return;
    }

    // read file, storing valid chars in imap
    char ch;
    while ((ch = map_file.get()) != EOF) {
        if (ch == '0' || ch == '1') {
            *imap = ch;
            ++imap;
        }
    }
    map_file.close();
}


/**
 * @brief Encode a character map to a uint8_t map.
 *
 * @param imap  Pointer to the input character map
 * @param emap  Pointer to the output uint8_t map
 */
void encode_map(const char* imap, std::uint8_t* emap) {
    // first row and last row
    const int t = N * (N - 1);
    for (int i = 0; i < N; ++i) {
        emap[i]     |= TOP;
        emap[t + i] |= BOT;
    }

    // left and right col
    for (int i = 0; i < NSQ; i += N) {
        emap[i]         |= LHS;
        emap[i + N - 1] |= RHS;
    }

    // obstacle
    for (int i = 0; i < NSQ; ++i) {
        if (imap[i] == '0') continue;

        emap[i] |= 0b11111;

        if (i >= N)
            emap[i - N] |= BOT;
        
        if (i < NSQ - N)
            emap[i + N] |= TOP;
        
        if (i % N != 0)
            emap[i - 1] |= RHS;
        
        if (i % N != (N - 1))
            emap[i + 1] |= LHS;
    }
}


/**
 * @brief Initialize an integer array with a default value, 
 *        and set the source and target node with 1 and -1.
 *
 * @param omap     Pointer to the integer array
 * @param src_id   ID of the source node
 * @param tgt_id   ID of the target node
 * @param len      Length of the array
 */
void init_omap(int* omap, const int src_id, const int tgt_id, const int len) {
    std::fill_n(omap, len, DEF_VAL);
    omap[src_id] =  1;
    omap[tgt_id] = -1;
}


/**
 * @brief Initialize the current and next queues.
 * 
 * @param curr_q Pointer to the current queue.
 * @param next_q Pointer to the next queue.
 * @param src_id Source cell ID for BFS.
 * @param tgt_id Target cell ID for BFS.
 */
void init_qs(int* curr_q, int* next_q, const int src_id, const int tgt_id) {
    curr_q[0] = src_id;
    curr_q[1] = ~tgt_id;
    curr_q[QID] = 2;
    next_q[QID] = 0;
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
 * @param emap    Pointer to the graph's edge map.
 * @param omap    Pointer to the output map.
 * @param curr_q  Pointer to the address of the current queue
 * @param next_q  Pointer to the address of the next queue
 *
 * @return  Elapsed time in ms.
 */
float measure_gpu_runtime(const std::uint8_t* emap, int* omap,
                          int** curr_q, int** next_q) {
    // event creation
    float gpu_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // start, running and synchronize
    cudaEventRecord(start);
    {
        bfs(emap, omap, curr_q, next_q);
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
 * @brief Rebuild the path from the target node to the source node 
 *        using the output map.
 *
 * @param omap    Pointer to the output map.
 * @param src_id  Source node ID.
 * @param tgt_id  Target node ID.
 * @param path    Vector of vectors representing the rebuilt path.
 */
void rebuild_path(const int* omap, const int src_id, const int tgt_id,
    std::vector<std::vector<int>>& path) {
    
    // build the source subpath
    int k = omap[SID];
    path.push_back({k / N, k % N});

    while (k != src_id) {
        k = omap[k];
        path.push_back({k / N, k % N});
    }
    // revserse path to let the source node in front
    std::reverse(path.begin(), path.end());

    // build the target subpath
    k = ~omap[TID];
    path.push_back({k / N, k % N});

    while (k != tgt_id) {
        k = ~omap[k];
        path.push_back({k / N, k % N});
    }
}


/**
 * @brief Measure the runtime of the path rebuilding process.
 *
 * @param omap    Pointer to the output map.
 * @param src_id  Source node ID.
 * @param tgt_id  Target node ID.
 * @param path    Vector of vectors representing the rebuilt path.
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
    char *imap;
    int  *omap;
    int  *curr_q;
    int  *next_q;
    std::uint8_t *emap;

    const int omap_size  = (NSQ * 2) + 2;
    const int q_capacity = (N * 4);

    cudaMallocManaged(&imap, NSQ * sizeof(char));
    cudaMallocManaged(&omap, omap_size * sizeof(int));
    cudaMallocManaged(&emap, NSQ * sizeof(std::int8_t));
    cudaMallocManaged(&curr_q, q_capacity * sizeof(int));
    cudaMallocManaged(&next_q, q_capacity * sizeof(int));

    // read and encode map
    read_map(map_file, imap);
    encode_map(imap, emap);
    
    // warm-up run
    init_omap(omap, src_ids[0], tgt_ids[0], omap_size);
    init_qs(curr_q, next_q, src_ids[0], tgt_ids[0]);
    measure_gpu_runtime(emap, omap, &curr_q, &next_q);

    float  gpu_time;
    double rbd_time;

    // display table header
    std::string hd_fmt = "%-4s%-11s%-15s%-15s\n";
    std::string bd_fmt = "%-4zu%-11.4f%-15.4lf%-15zu\n";
    printf(hd_fmt.c_str(), "No.", "gpu_time", "rebuild_time", "path_size");

    for (size_t i = 0; i < src_ids.size(); ++i) {
        // timing cuda pathfinding
        init_omap(omap, src_ids[i], tgt_ids[i], omap_size);
        init_qs(curr_q, next_q, src_ids[i], tgt_ids[i]);
        gpu_time = measure_gpu_runtime(emap, omap, &curr_q, &next_q);

        // timing cpu path rebuild
        std::vector<std::vector<int>> path;
        rbd_time = timing_path_rebuild(omap, src_ids[i], tgt_ids[i], path);
        
        // display results
        printf(bd_fmt.c_str(), i, gpu_time, rbd_time, path.size() - 1);
        write_path_to_csv(path, out_file, i);
    }
    
    // deallocation
    cudaFree(imap);
    cudaFree(emap);
    cudaFree(omap);
    cudaFree(curr_q);
    cudaFree(next_q);
}


/**
 * Organize data and run tests on the sparse map and the dense map.
*/
int main(int argc, char *argv[]) {
    // pathfinding on sparse map
    {
        const std::string map_file = "../maps/map_512_sparse.txt";
        const std::string out_file = "../out/path_S4_DW_queue";

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

    // pathfinding on dense map
    {
        const std::string map_file = "../maps/map_512_dense.txt";
        const std::string out_file = "../out/path_D4_DW_queue";
        
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