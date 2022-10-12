#  Phân tích thuật toán
#  Lab 2
#  Nguyễn Quốc Bảo - 18110053

import numpy as np
import random


count_compare = 0
count_assign = 0


# Recursive merge sort

def mergesort(array, left_index, right_index):   
    if left_index >= right_index:
        return

    middle = (left_index + right_index)//2
    mergesort(array, left_index, middle)
    mergesort(array, middle + 1, right_index)
    merge(array, left_index, middle, right_index)

# Merge Function
def merge(arr, l, m, r):
    global count_assign
    global count_compare

    n1 = m - l + 1
    n2 = r - m
    L = [0] * n1
    R = [0] * n2
    count_assign += 4

    for i in range(0, n1):
        L[i] = arr[l + i]
        count_compare += 1
        count_assign += 1 
    count_compare += 1
    
    for i in range(0, n2):
        R[i] = arr[m + i + 1]
        count_compare += 1
        count_assign += 1 
    count_compare += 1

    i, j, k = 0, 0, l
    count_compare += 2
    count_assign += 3

    while i < n1 and j < n2:
        count_compare += 2 
        if L[i] > R[j]:
            arr[k] = R[j]
            j += 1
            count_assign += 2 
        else:
            arr[k] = L[i]
            i += 1
            count_assign += 2 
        k += 1
        count_assign += 1 
    count_compare += 1
    
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1
        count_compare += 1
        count_assign += 3 
    count_compare += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1
        count_compare += 1
        count_assign += 3 
    count_compare += 1


def find_sumPair(A, N, x):
    global count_assign
    global count_compare
    mergesort(A, 0, len(A)-1)
    i = 0
    j = N - 1
    count_assign += 2 
    while(i < j):
        count_compare += 2 
        if (A[i] + A[j] == x):
            return i,j
        elif (A[i] + A[j] < x):
            i = i+1
            count_assign += 1 
        else:
            j = j - 1
            count_assign += 1
    count_compare += 1
    return -1,-1



if __name__ == '__main__':
    x = 50
    N = np.arange(10,1010,10)
    for n in N:
        count_compare = 0
        count_assign = 0
        A = np.random.randint(1, 10000, size = n)
        # print("Array A: \n",A)
        i,j = find_sumPair(A, len(A), x)
        # print("Sorted array \n",A)
        if (i != -1 and j != -1):
            print("Have pair ({}, {}) \n".format(i,j))
        else:
            print("Not exist \n")
        print("N: ",n)
        print("count_compare: {} and count_assign: {}".format(count_compare,count_assign))
