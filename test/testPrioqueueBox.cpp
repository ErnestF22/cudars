#include <cudars/prioqueue.h>

int main()
{
    // C++ code to implement Priority Queue using Linked List
    // Code found at: https://www.geeksforgeeks.org/priority-queue-using-linked-list/#

    // Create a Priority Queue
    cudars::Vec2d min = make_double2(7,4);
    cudars::Vec2d max = make_double2(5,6);
    double eps = 0.1;

    cudars::CuBox box;
    cudars::initCuBox(box, min, max, eps);

    cudars::NodeBox *pq;
    pq = cudars::newNodeBox(box);
    pushBox(&pq, box);
    pushBox(&pq, box);
    pushBox(&pq, box);

    while (!isEmptyBox(&pq))
    {
        auto peekbox = peekBox(&pq);
        std::cout << " " << peekbox.lower_;
        popBox(&pq);
    }
    std::cout << std::endl;

    return 0;
}

// This code is contributed by shivanisinghss2110