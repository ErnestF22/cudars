#ifndef CUDARS_PRIOQUEUE_CUH_
#define CUDARS_PRIOQUEUE_CUH_

#include <bits/stdc++.h>

#include <cudars/definitions.h>

#include <cudars/CuBox.cuh>

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
__host__ __device__ NodeBox *newNodeBox(cudars::CuBox box)
{
    NodeBox *temp = (NodeBox *)malloc(sizeof(NodeBox));

    // temp->box.min_ = box.min_;
    // temp->box.max_ = box.max_;
    // temp->box.lower_ = 0.0;
    // temp->box.upper_ = 0.0;
    // temp->box.eps_ = box.eps_;
    temp->box = box;

    temp->priority = box.lower_;
    temp->next = NULL;

    return temp;
}

// NodeBox *newNodeBox(Vec2d min, Vec2d max, double lower, double upper, double eps)
// {
//     NodeBox *temp = (NodeBox *)malloc(sizeof(NodeBox));

//     temp->box.min_ = min;
//     temp->box.max_ = max;
//     temp->box.lower_ = 0.0;
//     temp->box.upper_ = 0.0;
//     temp->box.eps_ = eps;

//     temp->priority = lower;
//     temp->next = NULL;

//     return temp;
// }

__host__ __device__ void computeBoundsInlier(cudars::Vec2d &min_, cudars::Vec2d &max_, double &lower_, double &upper_, double eps_,
                                             const cudars::VecVec2d &ptsSrc, const cudars::VecVec2d &ptsDst)
{
    // Vec2d mid = 0.5 * (min_ + max_);
    cudars::Vec2d mid = cudars::vec2sumWRV(min_, max_);
    cudars::scalarMul(mid, 0.5);
    cudars::Vec2d srcTransl;
    double dist, len;
    bool inlierFoundUpper, inlierFoundLower;

    // len = 0.5 * (max_ - min_).maxCoeff(); // Half of Infinity norm
    len = 0.5 * cudars::maxCoeffWRV(cudars::vec2diffWRV(max_, min_)); // Half of Infinity norm
    lower_ = (double)ptsSrc.size();
    upper_ = (double)ptsSrc.size();
    // ARS_VARIABLE4(lower_, upper_, len, mid);
    for (int is = 0; is < ptsSrc.size(); ++is)
    {
        // srcTransl = ptsSrc[is] + mid;
        cudars::vec2sum(srcTransl, ptsSrc[is], mid);
        inlierFoundLower = false;
        inlierFoundUpper = false;
        // ARS_VAR1(srcTransl.transpose());
        for (int id = 0; id < ptsDst.size() && !(inlierFoundLower && inlierFoundUpper); ++id)
        {
            // dist = (ptsDst[id] - srcTransl).norm();
            // dist = (ptsDst[id] - srcTransl).cwiseAbs().maxCoeff(); // Infinity norm
            dist = cudars::maxCoeffWRV(cudars::cwiseAbsWRV(cudars::vec2diffWRV(ptsDst[id], srcTransl))); // Infinity norm
            // ARS_VARIABLE4(ptsDst[id].transpose(), dist, dist < eps_, dist < eps_ + len);
            if (dist < eps_)
            {
                inlierFoundUpper = true;
            }
            if (dist < eps_ + len)
            {
                inlierFoundLower = true;
            }
        }
        if (inlierFoundLower)
            lower_ -= 1.0;
        if (inlierFoundUpper)
            upper_ -= 1.0;
    }
}

NodeBox *newNodeBox(NodeBox *queue,
                    cudars::CuBox newBox,
                    const cudars::VecVec2d &ptsSrc,
                    const cudars::VecVec2d &ptsDst)
{

    NodeBox *temp = (NodeBox *)malloc(sizeof(NodeBox));

    // temp->box.min_ = box.min_;
    // temp->box.max_ = box.max_;
    // temp->box.lower_ = 0.0;
    // temp->box.upper_ = 0.0;
    // temp->box.eps_ = box.eps_;
    temp->box = newBox;

    temp->priority = newBox.lower_;
    temp->next = NULL;

    queue = temp;

    computeBoundsInlier(temp->box.min_, temp->box.max_, temp->box.lower_, temp->box.upper_, temp->box.eps_,
                        ptsSrc, ptsDst);

    return queue;
}

__host__ __device__ void initCuBox(cudars::CuBox &box, const cudars::Vec2d &min, const cudars::Vec2d &max, const double eps)
{
    box.min_ = min;
    box.max_ = max;
    box.lower_ = 0.0;
    box.upper_ = 0.0;
    box.eps_ = eps;
}

__host__ __device__ void initCuBox(cudars::CuBox &box,
                                   const cudars::Vec2d &min,
                                   const cudars::Vec2d &max,
                                   const cudars::VecVec2d &ptsSrc,
                                   const cudars::VecVec2d &ptsDst,
                                   const double eps)
{
    // double dist, distMin, distUpper, distUpperMin;
    // Vec2d boxMin, boxMax, boxMid;
    box.min_ = min;
    box.max_ = max;
    box.eps_ = eps;
    // computeBoundsNaive(ptsSrc, ptsDst);
    NodeBox *nodeBox = newNodeBox(box);
    computeBoundsInlier(nodeBox->box.min_, nodeBox->box.max_, nodeBox->box.lower_, nodeBox->box.upper_, nodeBox->box.eps_, ptsSrc, ptsDst);
}

// void Box::computeBoundsNaive(const VecVec2d &ptsSrc,
//                              const VecVec2d &ptsDst)
// {
//     double distLower, distLowerMin, distUpper, distUpperMin;
//     Vec2d boxMin, boxMax, boxMid;
//     lower_ = 0.0;
//     upper_ = 0.0;
//     for (int is = 0; is < ptsSrc.size(); ++is)
//     {
//         distLowerMin = 1e+6;
//         distUpperMin = 1e+6;
//         // boxMin = ptsSrc[is] + min_;
//         // boxMax = ptsSrc[is] + max_;
//         // boxMid = ptsSrc[is] + 0.5 * (max_ + min_);
//         vec2sum(boxMin, ptsSrc[is], min_);
//         vec2sum(boxMax, ptsSrc[is], max_);
//         vec2sum(boxMid, ptsSrc[is], scalarMulWRV(vec2sumWRV(max_, min_), 0.5));
//         for (int id = 0; id < ptsDst.size(); ++id)
//         {
//             distLower = distancePointBox(ptsDst[id], boxMin, boxMax);
//             if (distLower < distLowerMin)
//             {
//                 distLowerMin = distLower;
//             }
//             // distUpper = (ptsDst[id] - (ptsSrc[is] + boxMid)).squaredNorm();
//             distUpper = vec2squarednorm(vec2diffWRV(ptsDst[id], vec2sumWRV(ptsSrc[is], boxMid)));
//             // ARS_VARIABLE5(boxMid.transpose(), ptsSrc[is].transpose(),
//             //          ptsDst[id].transpose(), distUpper, distUpperMin);
//             if (distUpper < distUpperMin)
//             {
//                 distUpperMin = distUpper;
//             }
//         }
//         lower_ += distLowerMin;
//         upper_ += distUpperMin;
//         // ARS_VARIABLE4(distLowerMin, distUpperMin, lower_, upper_);
//     }
// }

// Return the value at head
__host__ __device__ cudars::CuBox peekBox(NodeBox **head)
{
    return (*head)->box;
}

// Removes the element with the
// highest priority from the list
__host__ __device__ void popBox(NodeBox **head)
{
    NodeBox *temp = *head;
    (*head) = (*head)->next;
    free(temp);
}

// Function to push according to priority
__host__ __device__ void pushBox(NodeBox **head, cudars::CuBox box)
{
    NodeBox *start = (*head);

    // Create new Node
    NodeBox *temp = newNodeBox(box);

    // Special Case: The head of list has
    // lesser priority than new node. So
    // insert newnode before head node
    // and change head node.
    if ((*head)->priority > temp->priority)
    {

        // Insert New Node before head
        temp->next = *head;
        (*head) = temp;
    }
    else
    {

        // Traverse the list and find a
        // position to insert new node
        while (start->next != NULL &&
               start->next->priority < temp->priority)
        {
            start = start->next;
        }

        // Either at the ends of the list
        // or at required position
        temp->next = start->next;
        start->next = temp;
    }
}

// Function to check is list is empty
__host__ __device__ int isEmptyBox(NodeBox **head)
{
    return (*head) == NULL;
}

#endif /*CUDARS_PRIOQUEUE_CUH_*/