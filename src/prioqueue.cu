#include <cudars/prioqueue.cuh>

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
                                             cudars::Vec2d *ptsSrc, cudars::Vec2d *ptsDst, int ptsSrcSize, int ptsDstSize)
{
    // Vec2d mid = 0.5 * (min_ + max_);
    cudars::Vec2d mid;
    mid.x = 0.5 * (min_.x + max_.x);
    mid.y = 0.5 * (min_.y + max_.y);

    cudars::Vec2d srcTransl;
    double dist, len;
    bool inlierFoundUpper, inlierFoundLower;

    // len = 0.5 * (max_ - min_).maxCoeff(); // Half of Infinity norm
    len = 0.5 * max(max_.x - min_.x, max_.y - min_.y); // Half of Infinity norm
    // lower_ = (double)ptsSrc.size();
    // upper_ = (double)ptsSrc.size();
    lower_ = (double)ptsSrcSize;
    upper_ = (double)ptsDstSize;
    // ARS_VARIABLE4(lower_, upper_, len, mid);
    for (int is = 0; is < ptsSrcSize; ++is)
    {
        // srcTransl = ptsSrc[is] + mid;
        srcTransl.x = ptsSrc[is].x + mid.x;
        srcTransl.y = ptsSrc[is].y + mid.y;
        inlierFoundLower = false;
        inlierFoundUpper = false;
        // ARS_VAR1(srcTransl.transpose());
        for (int id = 0; id < ptsDstSize && !(inlierFoundLower && inlierFoundUpper); ++id)
        {
            // dist = (ptsDst[id] - srcTransl).norm();
            // dist = (ptsDst[id] - srcTransl).cwiseAbs().maxCoeff(); // Infinity norm
            // dist = cudars::maxCoeffWRV(cudars::cwiseAbsWRV(cudars::vec2diffWRV(ptsDst[id], srcTransl))); // Infinity norm
            dist = max(fabs(ptsDst[id].x - srcTransl.x), fabs(ptsDst[id].y - srcTransl.y));
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
                    cudars::Vec2d *ptsSrc,
                    cudars::Vec2d *ptsDst,
                    int ptsSrcSize,
                    int ptsDstSize)
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
                        ptsSrc, ptsDst, ptsSrcSize, ptsDstSize);

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
                                   cudars::Vec2d *ptsSrc,
                                   cudars::Vec2d *ptsDst,
                                   int ptsSrcSize,
                                   int ptsDstSize,
                                   const double eps)
{
    printf("Running initCuBox\n");
    // double dist, distMin, distUpper, distUpperMin;
    // Vec2d boxMin, boxMax, boxMid;
    box.min_ = min;
    box.max_ = max;
    box.eps_ = eps;
    // computeBoundsNaive(ptsSrc, ptsDst);
    NodeBox *nodeBox = newNodeBox(box);
    printf("Running computeBoundsInlier inside initCuBox\n");
    computeBoundsInlier(nodeBox->box.min_, nodeBox->box.max_, nodeBox->box.lower_, nodeBox->box.upper_, nodeBox->box.eps_, ptsSrc, ptsDst, ptsSrcSize, ptsDstSize);
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

__host__ __device__ void popBox(NodeBox **head)
{
    NodeBox *temp = *head;
    (*head) = (*head)->next;
    free(temp);
}

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

__host__ __device__ int isEmptyBox(NodeBox **head)
{
    printf("isEmptyBox int: %d\n", (*head) == NULL);
    return (*head) == NULL;
}

__host__ __device__ int getSizeBox(NodeBox **head) {
    int sz = 0;
    while ((*head)->next != NULL)
    {
        sz++;
        printf("queue size %d\n", sz);
        *head = (*head) -> next; 
    }
    return sz;
}