/**
 * @file          astar.h
 * @brief         Defines the function prototype `shortest_path()` and related 
 *                structs utilized in the A* algorithm.
 *
 * @author        Xin Cai
 * @email         xcai72@wisc.edu
 * @date          Nov. 24, 2023
 *
 * @course        ME759: High Performance Computing for Engineering Application
 * @instructor    Professor Dan Negrut
 * @assignment    Final Project
 */
#ifndef ASTAR_H
#define ASTAR_H

#include <vector>
#include <queue>
#include <cmath>

#define N 512

// Node struct of A*
struct Node {
    // field
    int x, y;        // current point on the map
    int accu_cost;   // accumulated cost, i.e., the actual cost
    int pred_cost;   // predicted cost, computed from heuristic
    int px, py;      // previous point
    
    // constructor with parameters
    Node(int x=-1, int y=-1, int a_cost=0, int p_cost=0, int px=-1, int py=-1)
        : x(x), y(y), accu_cost(a_cost), pred_cost(p_cost), px(px), py(py) {}
};

// comparator for node pointers to enable priority queue handling.
struct CompNodePtr {
    bool operator()(const Node* a, const Node* b) const {
        return (a->accu_cost + a->pred_cost) > (b->accu_cost + b->pred_cost);
    }
};

/**
 * @brief Find the shortest path between source and target points on a map 
 *        using A* algorithm.
 *
 * @param map The map layout represented as a 2D vector of characters.
 * @param source The coordinates of the source point.
 * @param target The coordinates of the target point.
 * @param m_type The type of map, specifying the allowed movement pattern.
 * @param openPQ The Open Set for the A* algorithm, containning unvisited node.
 * @param closed The Closed Set for the A* algorithm, containning visited node.
 * @param path Output vector containing the coordinates of the shortest path.
 * @param loopcounter The number of iterations performed by the A* algorithm.
 */
void shortest_path(
    const std::vector<std::vector<char>> &map,
    const std::vector<int> source,
    const std::vector<int> target,
    const int m_type,
    std::priority_queue<Node*, std::vector<Node*>, CompNodePtr>& openPQ,
    std::vector<std::vector<Node*>>& closed,
    std::vector<std::vector<int>>& path,
    int& loopcounter);

#endif
