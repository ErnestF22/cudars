#include <cudars/prioqueue.h>

namespace cudars
{

    void prioqueue::create_queue()
    {
        front = rear = -1;
    }
    void prioqueue::insert_element(int data)
    {
        if (rear >= MAX_PRIOQUEUE_SIZE - 1)
        {
            printf("\nQUEUE OVERFLOW");
            return;
        }
        if ((front == -1) && (rear == -1))
        {
            front++;
            rear++;
            pqueue[rear] = data;
            return;
        }
        else
            check_priority(data);
        rear++;
    }
    void prioqueue::check_priority(int data)
    {
        int i, j;
        for (i = 0; i <= rear; i++)
        {
            if (data >= pqueue[i])
            {
                for (j = rear + 1; j > i; j--)
                {
                    pqueue[j] = pqueue[j - 1];
                }
                pqueue[i] = data;
                return;
            }
        }
        pqueue[i] = data;
    }
    void prioqueue::delete_element(int data)
    {
        int i;
        if ((front == -1) && (rear == -1))
        {
            printf("\nEmpty Queue");
            return;
        }
        for (i = 0; i <= rear; i++)
        {
            if (data == pqueue[i])
            {
                for (; i < rear; i++)
                {
                    pqueue[i] = pqueue[i + 1];
                }
                pqueue[i] = -99;
                rear--;
                if (rear == -1)
                    front = -1;
                return;
            }
        }
        printf("\n%d element not found in queue", data);
    }
    void prioqueue::display_priorityqueue()
    {
        if ((front == -1) && (rear == -1))
        {
            printf("\nEmpty Queue ");
            return;
        }
        for (; front <= rear; front++)
        {
            printf(" %d ", pqueue[front]);
        }
        front = 0;
    }

}