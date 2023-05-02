#include <cudars/prioqueue.h>

// Driver code
int main()
{
    // C++ code to implement Priority Queue using Linked List
    // Code found at: https://www.geeksforgeeks.org/priority-queue-using-linked-list/#

    // Create a Priority Queue
    // 7->4->5->6
    cudars::Node *pq = cudars::newNode(4, 1);
    push(&pq, 5, 2);
    push(&pq, 6, 3);
    push(&pq, 7, 0);

    while (!isEmpty(&pq))
    {
        std::cout << " " << peek(&pq);
        pop(&pq);
    }
    std::cout << std::endl;

    return 0;
}

// This code is contributed by shivanisinghss2110