#ifndef CUDARS_PRIOQUEUE_CUH_
#define CUDARS_PRIOQUEUE_CUH_

#include <bits/stdc++.h>

#include <cudars/definitions.h>
#include <cudars/utils.h>

// #include <cudars/CuBox.cuh>

#define MAX_PRIOQUEUE_SIZE 1024

// NodeBox
struct NodeBox
{
    cudars::CuBox box;

    // Lower values indicate
    // higher priority
    int priority;

    NodeBox *next;
};

// auto cmp = [](const Box &box1, const Box &box2)
// {
//     return box1.lower_ > box2.lower_;
// };
// std::priority_queue<Box, std::vector<Box>, decltype(cmp)> prioqueue(cmp);

// Function to create a new node
__host__ __device__ NodeBox *newNodeBox(cudars::CuBox box);

__host__ __device__ void computeBoundsInlier(cudars::Vec2d &min_, cudars::Vec2d &max_, double &lower_, double &upper_, double eps_,
                                             cudars::Vec2d *ptsSrc, cudars::Vec2d *ptsDst, int ptsSrcSize, int ptsDstSize);

NodeBox *newNodeBox(NodeBox *queue,
                    cudars::CuBox newBox,
                    cudars::Vec2d *ptsSrc,
                    cudars::Vec2d *ptsDst,
                    int ptsSrcSize,
                    int ptsDstSize);

__host__ __device__ void initCuBox(cudars::CuBox &box, const cudars::Vec2d &min, const cudars::Vec2d &max, const double eps);

__host__ __device__ void initCuBox(cudars::CuBox &box,
                                   const cudars::Vec2d &min,
                                   const cudars::Vec2d &max,
                                   cudars::Vec2d *ptsSrc,
                                   cudars::Vec2d *ptsDst,
                                   int ptsSrcSize,
                                   int ptsDstSize,
                                   const double eps);

// Return the value at head without removing it
__host__ __device__ cudars::CuBox peekBox(NodeBox **head);

// Removes the element with the
// highest priority from the list
__host__ __device__ void popBox(NodeBox **head);

// Function to push according to priority
__host__ __device__ void pushBox(NodeBox **head, cudars::CuBox box);

// Function to check is list is empty
__host__ __device__ int isEmptyBox(NodeBox **head);

#endif /*CUDARS_PRIOQUEUE_CUH_*/
