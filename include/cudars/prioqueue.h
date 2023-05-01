#ifndef CUDARS_PRIOQUEUE_H
#define CUDARS_PRIOQUEUE_H

#include <stdio.h>
#include <stdlib.h>

#define MAX_PRIOQUEUE_SIZE 1024

namespace cudars
{
    struct prioqueue
    {
        int front;
        int rear;

        int pqueue[MAX_PRIOQUEUE_SIZE];

        void create_queue();
        void insert_element(int);
        void delete_element(int);
        void check_priority(int);
        void display_priorityqueue();
    };
}

#endif /*CUDARS_PRIOQUEUE_H*/
