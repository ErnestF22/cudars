#include <iostream>

#include <cudars/prioqueue.h>

int main(int argc, char **argv) 
{ 
    int n, choice;  
    printf("\nEnter 1 to insert element by priority "); 
    printf("\nEnter 2 to delete element by priority "); 
    printf("\nEnter 3 to display priority queue "); 
    printf("\nEnter 4 to exit");  
    cudars::prioqueue pq;
    pq.create_queue();  
    while (1) 
    { 
        printf("\nEnter your choice : ");    
        scanf("%d", &choice);   
        switch(choice) 
        { 
        case 1: 
            printf("\nEnter element to insert : "); 
            scanf("%d",&n); 
            pq.insert_element(n); 
            break; 
        case 2: 
            printf("\nEnter element to delete : "); 
            scanf("%d",&n); 
            pq.delete_element(n); 
            break; 
        case 3: 
            pq.display_priorityqueue(); 
            break; 
        case 4: 
            exit(0); 
        default: 
            printf("\n Please enter valid choice"); 
        } 
    } 
}  