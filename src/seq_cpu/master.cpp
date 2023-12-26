/**
 * @file          master.cpp
 * @brief         This program employs the A* algorithm on two predefined maps
 *                in a single-core CPU environment. It conducts 10 pathfinding 
 *                tasks, measures the execution time, and outputs the results 
 *                as CSV files.
 *
 * @author        Xin Cai
 * @email         xcai72@wisc.edu
 * @date          Nov. 24, 2023
 *
 * @course        ME759: High Performance Computing for Engineering Application
 * @instructor    Professor Dan Negrut
 * @assignment    Final Project
 *
 * @module        none
 * @compile       g++ master.cpp astar.cpp -Wall -O3 -std=c++17 -o pathfinding
 */
#include "astar.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <chrono>


/**
 * @brief Reads a map from a file and populates a 2D vector.
 *
 * @param filename The name of the file containing the map
 * @param map      A 2D vector to store the map data
 */
void read_map(std::string filename, std::vector<std::vector<char>>& map) {
    std::ifstream map_file(filename);
    
    if (!map_file.is_open()) {
        std::cout << "ERROR: fail to open \'" << filename << "\'" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(map_file, line)) {
        std::vector<char> row;
        for (char ch : line) {
            if (ch == '0' || ch == '1') {
                row.push_back(ch);
            }
        }
        map.push_back(row);
    }
    map_file.close();
}


/**
 * @brief Writes a path to a CSV file.
 *
 * @param path      The path to be written to the CSV file
 * @param out_file  The base name of the output CSV file
 * @param id        An identifier used in the filename
 */
void write_path_to_csv(const std::vector<std::vector<int>>& path, 
                       const std::string out_file, const int id) {
    // build complete file path
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
 * @brief Safely clears resources from priority queue and 2D vector of Nodes.
 *
 * @param openPQ Priority queue of Node pointers to release
 * @param closed 2D vector of Node pointers to release
 */
void safe_clear(
    std::priority_queue<Node*, std::vector<Node*>, CompNodePtr>& openPQ,
    std::vector<std::vector<Node*>>& closed) {
    // release Nodes in openPQ
    Node* K;
    while (!openPQ.empty()) {
        K = openPQ.top();
        openPQ.pop();
        delete K;
    }
    // release Nodes in closed
    for (size_t i = 0; i < closed.size(); ++i) {
        for (size_t j = 0; j < closed[i].size(); ++j) {
            if (closed[i][j] != nullptr) {
                delete closed[i][j];
                closed[i][j] = nullptr;
            }
        }
    } 
}


/**
 * @brief Measures the CPU runtime of the shortest_path function.
 *
 * @param map      Input map for pathfinding
 * @param source   Source point
 * @param target   Target point
 * @param m_type   Map type
 * @param openPQ   Priority queue for node pointers in the pathfinding process
 * @param closed   2D vector of node pointers for closed set
 * @param path     2D vector storing the computed path
 * @param loop_n   Output parameter for the loop count
 * @param time_ms  Output parameter for the measured CPU runtime in ms
 */
void measure_runtime(
    const std::vector<std::vector<char>> &map,
    const std::vector<int> source,
    const std::vector<int> target,
    const int m_type,
    std::priority_queue<Node*, std::vector<Node*>, CompNodePtr>& openPQ,
    std::vector<std::vector<Node*>>& closed,
    std::vector<std::vector<int>>& path,
    int& loop_n,
    double& time_ms) {
    
    // prepare timing tools
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration_ms;

    // measure the execution time
    start = std::chrono::high_resolution_clock::now();
    { 
       shortest_path(map, source, target, m_type, 
                     openPQ, closed, path, loop_n);
    }
    end = std::chrono::high_resolution_clock::now();

    // compute duration
    using dur_in_ms = std::chrono::duration<double, std::milli>;
    duration_ms = std::chrono::duration_cast<dur_in_ms>(end - start);
    time_ms = duration_ms.count();
}


/**
 * @brief Runs pathfinding tests and outputs results to CSV files.
 *
 * @param map_file  The file containing the map for pathfinding
 * @param out_file  The base name for output CSV files
 * @param m_type    The type parameter for the pathfinding algorithm
 * @param src_ids   Vector of source points for pathfinding tasks
 * @param tgt_ids   Vector of target points for pathfinding tasks
 */
void run_test(
    const std::string map_file,
    const std::string out_file,
    const int m_type,
    const std::vector<int>& src_ids,
    const std::vector<int>& tgt_ids) {
    
    const std::string separator(40, '-');
    printf("%s\n", separator.c_str());
    printf("map file: \'%s\'\n\n", map_file.c_str());

    // read map
    std::vector<std::vector<char>> map;
    read_map(map_file, map);
    
    // declare input and output variables
    std::vector<std::vector<int>> path;
    std::vector<int> source;
    std::vector<int> target;
    int loop_n;
    double cpu_time;

    // warm-up run
    {
        std::priority_queue<Node*, std::vector<Node*>, CompNodePtr> openPQ;
        std::vector<std::vector<Node*>> closed(
            map.size(), 
            std::vector<Node*>(map[0].size(), nullptr));

        source = {src_ids[0] / N, src_ids[0] % N};
        target = {tgt_ids[0] / N, tgt_ids[0] % N};

        measure_runtime(map, source, target, m_type, openPQ, 
                        closed, path, loop_n, cpu_time);

        safe_clear(openPQ, closed);
        path.clear();
    }

    // display table header
    std::string hd_fmt = "%-4s%-11s%-11s%-15s\n";
    std::string bd_fmt = "%-4zu%-11.4f%-11.4d%-15zu\n";
    printf(hd_fmt.c_str(), "No.", "cpu_time", "loop_size", "path_size");

    // run real tests
    for (size_t i = 0; i < src_ids.size(); ++i) {
        // prepare parameters
        std::priority_queue<Node*, std::vector<Node*>, CompNodePtr> openPQ;
        std::vector<std::vector<Node*>> closed(
            map.size(), 
            std::vector<Node*>(map[0].size(), nullptr));

        source = {src_ids[i] / N, src_ids[i] % N};
        target = {tgt_ids[i] / N, tgt_ids[i] % N};

        // timing execution
        measure_runtime(map, source, target, m_type, openPQ, 
                        closed, path, loop_n, cpu_time);

        // display results
        printf(bd_fmt.c_str(), i, cpu_time, loop_n, path.size() - 1);
        write_path_to_csv(path, out_file, i);

        safe_clear(openPQ, closed);
        path.clear();
    }
}


/**
 * Organize data and run tests on the sparse map and the dense map.
*/
int main(int argc, char *argv[]) {
    // run tests on sparse map
    {
        const std::string map_file = "../maps/map_512_sparse.txt";
        const std::string ot4_file = "../out/path_S4_cpu";
        const std::string ot6_file = "../out/path_S6_cpu";

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

        run_test(map_file, ot4_file, 4, src_ids, tgt_ids);
        run_test(map_file, ot6_file, 6, src_ids, tgt_ids);
    }

    // run tests on dense map
    {
        const std::string map_file = "../maps/map_512_dense.txt";
        const std::string ot4_file = "../out/path_D4_cpu";
        const std::string ot6_file = "../out/path_D6_cpu";
        
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

        run_test(map_file, ot4_file, 4, src_ids, tgt_ids);
        run_test(map_file, ot6_file, 6, src_ids, tgt_ids);
    }
    return 0;
}
