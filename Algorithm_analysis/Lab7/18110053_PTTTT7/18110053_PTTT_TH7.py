#  Phân tích thuật toán
# Bài tập thực hành tuần 5
#  Nguyễn Quốc Bảo - 18110053

#=========================== Import ===========================

import numpy as np
from random import sample
from collections import Counter
from itertools import permutations

#=========================== function Exercise 1===========================
# frequency of elements is more than n/k
def findmajorityNbyK(arr, n, k) :
    global counter_assign
    global counter_compare   
    
    x = n // k
      
    # map initialization
    freq = {}
    counter_assign += 2


    for i in range(n) :
        counter_compare += 2   
        if arr[i] in freq :
            freq[arr[i]] += 1
            counter_assign += 1
        else :
            freq[arr[i]] = 1
            counter_assign += 1

    # Store result
    ls_result = []    
    # Traversing the map
    for i in freq :
        counter_compare += 2
        # Checking if value of a key-value pair
        # is greater than x (where x=n/k)
        if (freq[i] >= x) :
              
            # append the key of whose value
            # is greater than x
            ls_result.append(i)

    return ls_result



def ex1():
    global counter_assign
    global counter_compare
    counter_assign = 0
    counter_compare = 0 
    print('-'*40,'Exercise 1','-'*40)
    N = int(input('>> Input number of elements array random : '))
    print('initializing random array from 0 to 100:')
    A = np.random.randint(0,100,N)
    n = len(A)
    print('A = ')
    print(A)
    result = findmajorityNbyK(A, len(A), 10)
    print('Elements appear at least n/10 = {} times: '.format(n//10))
    print(result)
    print("When N = {} , then count_compare: {} and count_assign: {}".format(n, counter_compare,counter_assign))
    print('#'*40,'End Exercise 1','#'*40)
    pass


#=========================== function Exercise 2===========================

def mergeArrays(arr1, arr2, n1, n2):
    global counter_assign
    global counter_compare

    arr3 = [None] * (n1 + n2)
    i = 0
    j = 0
    k = 0

    counter_assign += 4

    # Traverse both array
    while i < n1 and j < n2:
        counter_compare += 2

        # Check if current element of first array is smaller
        # than current element of second array.
        # If yes, store first array element and increment first array index.
        # Otherwise do same with second array
        if arr1[i] < arr2[j]:
            arr3[k] = arr1[i]
            k = k + 1
            i = i + 1
            counter_assign += 3
        else:
            arr3[k] = arr2[j]
            k = k + 1
            j = j + 1
            counter_assign += 3
     
 
    # Store remaining elements
    # of first array
    while i < n1:
        counter_compare += 1
        counter_assign += 3

        arr3[k] = arr1[i]
        k = k + 1
        i = i + 1
 
    # Store remaining elements
    # of second array
    while j < n2:
        counter_compare += 1
        counter_assign += 3

        arr3[k] = arr2[j]
        k = k + 1
        j = j + 1

    return arr3


def ex2():
    global counter_assign
    global counter_compare
    counter_assign = 0
    counter_compare = 0
    
    print('-'*40,'Exercise 2','-'*40) 
    N = int(input('>> Input number of elements array random : '))
    print('initializing random array from 0 to 100:')
    A = sample(range(100), k = N)
    n1 = len(A)
    B = sample(range(100), k = N)
    n2 = len(A)
    A.sort()
    B.sort()
    print('A = ')
    print(A)
    print('B = ')
    print(B)
    print()
    print("Array after merging A and B:")
    result = mergeArrays(A,B,n1,n2)
    s1 = ', '.join([str(value) for value in result[:(n1+n2)//2]])
    s2 = ', '.join([str(value) for value in result[(n1+n2)//2:]])
    print('[',s1+',\n',s2,']')
    print()
    print("When total N = {} , then count_compare: {} and count_assign: {}".format(n1+n2, counter_compare,counter_assign))
    print('#'*40,'End Exercise 2','#'*40)
    pass


#============================= main =============================

if __name__ == '__main__':
    print('\n')
    ex1()
    print('\n \n')
    ex2()