#  Phân tích thuật toán
# Bài tập thực hành tuần 5
#  Nguyễn Quốc Bảo - 18110053

#=======================================================

import numpy as np
from random import sample
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

#=======================================================

# O(n)
def Find_Subarrays_Given_Sum(array_, S):
    ''' Function to find subarrays with the given sum in a array '''
    global counter_assign
    global counter_compare    
    
    dict = {}

    dict.setdefault(0,[]).append(-1)

    curr_sum = 0

    counter_assign += 3
    result = []
    for index in range(len(array_)):

        counter_assign += 2
        counter_compare += 2

        curr_sum += array_[index]

        # check if there exists at least one subarray with the given S
        if (curr_sum - S) in dict:
                result = [array_[value +1: index+1] for value in dict.get(curr_sum - S)]
                return result

        # insert (curr_sum, index) pair into the dictionary
        dict.setdefault(curr_sum, []).append(index)

    return result

def test_ex1():
    print('-'*35,'Test demo','-'*35)
    global counter_assign
    global counter_compare
    S = 200
    N = np.arange(50,500+1,50)
    for n in N:
        counter_assign = 0
        counter_compare = 0
        A = sample(range(500), k=n)
        print('A = ',A)
        # subset = Knapsack(A,S)
        print()
        results = Find_Subarrays_Given_Sum(A,S)
        #If we reach here, then no subarray exists
        if (len(results) == 0):
            print("No subarray with given sum exists is {}".format(S))
        else:
            print('Subarrays sum to {}'.format(S))
            for R in results:
                print(R, end='\t') 
        print()
        print("When N = {} , then count_compare: {} and count_assign: {}".format(n, counter_compare,counter_assign))
        print('-'*80)

def ex1():
    print('-'*35,'Exercise 1','-'*35)
    global counter_assign
    global counter_compare

    counter_assign = 0
    counter_compare = 0

    # Input array and S
    S = int(input('>> Input S: '))
    print('Input array A ')
    # number of elements
    n = int(input(">> Enter number of elements : "))
    
    # Below line read inputs from user using map() function 
    A = list(map(int,input("\n>> Enter the numbers : ").strip().split()))[:n]
    results = Find_Subarrays_Given_Sum(A,S)
    #If we reach here, then no subarray exists
    if (len(results) == 0):
        print("No subarray with given sum exists is {}".format(S))
    else:
        print('Subarrays sum to {}'.format(S))
        for R in results:
            print(R, end='\t')
    print() 
    print("When N = {} , then count_compare: {} and count_assign: {}".format(n, counter_compare,counter_assign))
    print('-'*80)



#========================================================

if __name__ == '__main__':
    # ex1()
    test_ex1()



