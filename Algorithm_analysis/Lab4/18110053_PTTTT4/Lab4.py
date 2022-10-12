# Phân Tích Thuật Toán
# Bài tập thực hành tuần 4
# Nguyễn Quốc Bảo - 18110053

#------------------------------------------------------------------------------------------
import numpy as np
from random import sample
from numpy.core.defchararray import find, index

# ---------------------------------Exercise 1-----------------------------------------
counter_assign = 0
counter_compare = 0


# Binary search algorithm implementation
def binary_search(arr,left,right,x):
    if right >= left:
        mid = left + (right - left)//2

        if (arr[mid] == x):
            return mid
 
        if (arr[mid] > x):
            return binary_search(arr,left,mid-1,x)
 
        return binary_search(arr,mid+1,right,x)
 
    return -1

def test_ex1():
    N = np.arange(10,100,10)     
    for n in N:
        A = sorted(np.random.randint(100,size=n))
        x = np.random.randint(0,100)
        print('-'*80)
        print('Array sorted \n',A)
        print('N = ', n)
        print('x = ',x)
        index = binary_search(A, 0, len(A)-1, x)
        if (index != -1):
            print('found element {} at {}'.format(x,index))
        else:
            print('Not found')

def ex1():
    print('-'*35,'Exercise 1','-'*35,'\n')
    x = int(input(">> Input x :"))
    print("Input array A ")
    # number of elements
    n = int(input(">> Enter number of elements : "))
    
    # Below line read inputs from user using map() function 
    A = list(map(int,input("\n>> Enter the numbers : ").strip().split()))[:n]
    # Sort array
    A.sort()
    print('Array sorted \n',A)
    # find x in array by binary search
    index = binary_search(A, 0, len(A)-1, x)
    if (index != -1):
        print('found element {} at {}'.format(x,index))
    else:
        print('Not found')
    print()
    print('-'*20,'Test data','-'*20,'\n')
    test_ex1()

    print('\t\t\t END EX1 \n')


# ---------------------------------Exercise 2-----------------------------------------

def heapify(arr, n, i):
    global counter_assign 
    global counter_compare

    # Find largest among root, left child and right child
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    counter_assign +=3
    
    if l < n and arr[i] < arr[l]:
        largest = l
        counter_assign +=1
        
    if r < n and arr[largest] < arr[r]:
        largest = r
        counter_assign +=1
        
    # If root is not largest, swap with largest and continue heapifying
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        counter_assign +=1
        heapify(arr, n, largest)
    
    counter_compare += 3


def heapSort(arr):
    global counter_assign 
    global counter_compare    

    n = len(arr)

    # Build max heap
    for i in range(n//2, -1, -1):
        counter_compare +=1
        heapify(arr, n, i)

    for i in range(n-1, 0, -1):
        # Swap
        arr[i], arr[0] = arr[0], arr[i]
        counter_assign +=1
        counter_compare +=1
        # Heapify root element
        heapify(arr, i, 0)
# Function find k'th smallest element in a given array
def kth_smallest(arr, k):
    global counter_assign 
    global counter_compare

    # Sort the given array
    heapSort(arr)
 
    # Return k'th element in the
    # sorted array
    return arr[k-1]

def ex2():
    print('-'*35,'Exercise 2','-'*35,'\n')
    global counter_assign 
    global counter_compare
    # init N and k
    N = np.arange(10,100,10)
    k = 5

    # Run loop and create array random have size n, find k'th smallest
    for n in N:
        counter_assign = 0
        counter_compare = 0
        S = sample(range(1,1001), n)
        e = kth_smallest(S,k)
        print('-'*80)
        print('Array sorted \n',S)
        print("N = ",n)
        print('Element {} is {}th smallest '.format(e,k))
        print("count_compare: {} and count_assign: {}".format(counter_compare,counter_assign))

    print('\t\t\t END EX2 \n')
        


if __name__ == '__main__':
    ex1()
    ex2()