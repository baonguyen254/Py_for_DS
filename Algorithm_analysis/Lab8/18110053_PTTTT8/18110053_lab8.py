#  Phân tích thuật toán
#  Bài tập thực hành tuần 8
#  Nguyễn Quốc Bảo - 18110053

#=========================== Import ===========================

import numpy as np
from random import sample
from collections import Counter
from itertools import permutations

#=========================== function Exercise 1===========================


def Fibonnaci_char(n):
    global counter_assign
    global counter_compare

    f0 = 'abc'
    f1 = 'def'
    
    counter_assign += 2
    counter_compare += 1
    
    if (n == 1):
        return f1

    for i in range(0,n):

        f = f1 + f0 
        f0 = f1
        f1 = f
        counter_compare += 1
        counter_assign += 3

    return f0

def find_char_kth_fn(n,k):
    global counter_assign
    global counter_compare

    fn = Fibonnaci_char(n)
    print('f{} = {}'.format(n,fn))
    print('character '+str(k)+'th in f{} is'.format(n) , fn[k-1])

def ex1():
    global counter_assign
    global counter_compare
    counter_assign = 0
    counter_compare = 0
    n = 4; k = 3
    find_char_kth_fn(n,k)
    print("When N = {} , then count_compare: {} and count_assign: {}".format(n, counter_compare,counter_assign))

#============================= main =============================

if __name__ == '__main__':
    print('\n')
    ex1()
    print('\n')
    # ex2()