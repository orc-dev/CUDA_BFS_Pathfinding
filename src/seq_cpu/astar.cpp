/**
 * @file          astar.cpp
 * @brief         Implement the single-core CPU version of the A* algorithm 
 *                for the specified map settings outlined in the project.
 *
 * @author        Xin Cai
 * @email         xcai72@wisc.edu
 * @date          Nov. 24, 2023
 *
 * @course        ME759: High Performance Computing for Engineering Application
 * @instructor    Professor Dan Negrut
 * @assignment    Final Project
 */
#include "astar.h"
#include <iostream>

/**
 * @brief Calculate the heuristic value between two points on a map.
 *
 * This function computes the heuristic value based on the Manhattan distance
 * or a modified distance for different map types.
 *
 * @param x0 The x-coordinate of the first point.
 * @param y0 The y-coordinate of the first point.
 * @param x1 The x-coordinate of the second point.
 * @param y1 The y-coordinate of the second point.
 * @param m_type The type of map, specifying the allowed movement pattern
 *
 * @return The computed heuristic value.
 */
int heuristic(const int x0, const int y0, 
              const int x1, const int y1, const int m_type) {
    
    const int dx = std::abs(x0 - x1);
    const int dy = std::abs(y0 - y1);

    switch (m_type) {
        case 4:
            return dx + dy;
        
        case 6:
            return dx + std::max(2 + dy - dx, 0);

        case 8:
            return std::max(dx, dy);

        default:
            return 0;
    }
}


/**
 * @brief Get valid neighboring nodes based on the given node and relating
 *  map conditions.
 *
 * @param map The map layout represented as a 2D vector of characters.
 * @param closed A 2D vector indicating whether nodes have been processed.
 * @param target The coordinates of the target destination.
 * @param K The current node for which neighbors are being determined.
 * @param m_type The type of map, specifying the allowed movement pattern.
 * @param neibs A vector to store valid neighboring nodes.
 */
void get_valid_neibs(
    const std::vector<std::vector<char>>& map,
    const std::vector<std::vector<Node*>>& closed,
    const std::vector<int> target,
    const Node* K,
    const int m_type,
    std::vector<Node*>& neibs) {
    
    // compute coordinates of the adjacent points defined in MapType
    std::vector<std::vector<int>> points({
        {K->x, K->y - 1},
        {K->x, K->y + 1},
        {K->x + 1, K->y},
        {K->x - 1, K->y},
    });

    if (m_type == 6) {
        const int offset = ((K->x & 1) == 1) ? 1 : -1;
        points.push_back({K->x - 1, K->y + offset});
        points.push_back({K->x + 1, K->y + offset});
    }

    if (m_type == 8) {
        points.push_back({K->x - 1, K->y - 1});
        points.push_back({K->x - 1, K->y + 1});
        points.push_back({K->x + 1, K->y - 1});
        points.push_back({K->x + 1, K->y + 1});
    }

    // wrap points to Nodes
    const int xlim = map.size();
    const int ylim = map[0].size();

    for (const auto& p : points) {
        // must be within bounds
        // must be an open cell
        // must have not been processed
        if (p[0] >= 0 && p[0] < xlim && 
            p[1] >= 0 && p[1] < ylim &&
            map[p[0]][p[1]] == '0'   && 
            closed[p[0]][p[1]] == nullptr) {
            
            Node* ptr = new Node(
                p[0], p[1], 
                K->accu_cost + 1,
                heuristic(p[0], p[1], target[0], target[1], m_type),
                K->x, K->y
            );
            neibs.push_back(ptr);
        }
    }
}


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
    int& loopcounter) {
    
    // build source nodes
    Node* S = new Node(source[0], source[1], 0);
    Node* K = new Node(source[0], source[1], 0);

    openPQ.push(S);
    int counter = 0;

    // A-star processing
    while (!openPQ.empty()) {
        // get current Node for processing
        K = openPQ.top();
        openPQ.pop();
        
        // has been processed
        if (closed[K->x][K->y] != nullptr)
            continue;

        counter++;
        closed[K->x][K->y] = K;
        
        // hit the target
        if (K->x == target[0] && K->y == target[1])
            break;
        
        // get all valid neighbor nodes
        std::vector<Node*> neibs;
        get_valid_neibs(map, closed, target, K, m_type, neibs);

        for (auto& n_ptr : neibs)
            openPQ.push(n_ptr);
    }
    // update loopcounter
    loopcounter = counter;

    // reconstruct the shortest path
    path.push_back({K->x, K->y});

    while (K->x != S->x || K->y != S->y) {
        path.push_back({K->px, K->py});
        K = closed[K->px][K->py];
    }
}
