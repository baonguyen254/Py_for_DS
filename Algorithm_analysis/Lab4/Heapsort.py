import numpy as np
from numpy.core.defchararray import find, index


def heapify(arr, n, i):
    # Find largest among root and children
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and arr[i] < arr[l]:
        largest = l

    if r < n and arr[largest] < arr[r]:
        largest = r

    # If root is not largest, swap with largest and continue heapifying
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def heapSort(arr):
    n = len(arr)

    # Build max heap
    for i in range(n//2, -1, -1):
        heapify(arr, n, i)

    for i in range(n-1, 0, -1):
        # Swap
        arr[i], arr[0] = arr[0], arr[i]

        # Heapify root element
        heapify(arr, i, 0)



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

def ex1():
    print(">> Input x :")
    print(">> Input array A :")
    N = np.arange(10,100,10)
         
    for n in N:
        A = sorted(np.random.randint(100,size=n))
        x = np.random.randint(0,100)
        print('------------------------------------------------------')
        heapSort(A)
        print(A)
        print('N = ', n)
        print('x = ',x)
        index = binary_search(A, 0, len(A)-1, x)
        if (index != -1):
            print('found element {} at {}'.format(x,index))
        else:
            print('Not found')


if __name__ == '__main__':
    ex1()