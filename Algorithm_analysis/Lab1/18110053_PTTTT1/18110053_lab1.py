#  Phân tích thuật toán
#  Lab 1
#  Nguyễn Quốc Bảo - 18110053

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# -------------------Bai 1-----------------------
def decimal_to_binary(num):
    count_compare = 0
    count_assign = 0

    Bin = 0
    temp = num
    i = 1
    count_assign += 3
    while ( temp > 0):
        count_compare += 1
        Bin = Bin + (temp % 2)*i
        temp = temp // 2
        i = i*10
        count_assign += 3
    count_compare += 1
    return Bin, count_compare , count_assign


def Bai1_1():
    print('Câu 1')
    N = int(input('Nhập N: '))
    
    Bin,_, _ = decimal_to_binary(N)
    print('convert {} to Bin: {}'.format(N, Bin))


def Bai1_2():
    print('Câu 2')
    X = np.arange(1, 1001,1)
    Log2 = np.log2(X)
    ls_compare = []
    ls_assign = []
    for N in X:
        Bin, compare, assign = decimal_to_binary(N)
        ls_assign.append(assign)
        ls_compare.append(compare)
        print('Count step assign: {} and compare: {} of Binary({}) = {}'.format(assign,compare,N, Bin))

    plt.plot(Log2, color = 'blue', marker ='*')
    plt.plot(ls_assign, color = 'red', marker ='*')
    plt.plot(ls_compare, color = 'green', marker ='*')
    plt.legend(['Log2','assigns','compare'])
    plt.title('plot complex Log2')
    plt.show()

# ------------------- Bai 2 ------------------------
def Bonus(A, a, b, k, n):
    count_assign = 0
    count_compare = 0
    
    Sum = 0
    counter = defaultdict(int)
    i = 1
    j = 1
    count_assign += 4
    while i < n:
        count_compare += 2
        if A[i] <= b and A[i] >= a:
            counter[A[i]] += 1
            count_assign +=1
        i += 1
        count_assign +=1
    count_compare += 1
    while j  < k:
        count_compare += 2
        if j < k:
            Sum = Sum + j
            count_assign += 1
        j += 1
        count_assign += 1
    count_compare +=1
    return count_assign, count_compare

def K_fixed(a, b, k = 100):
    ls_compare = []
    ls_assign = []
    N = np.arange(10,10010,10)
    for n in N:
        A = np.random.randint(1,k,size=n)
        assign, compare = Bonus(A, a, b, k, n)
        ls_assign.append(assign)
        ls_compare.append(compare)
        print('When k fixed 100, N: {} '.format(n))       
        print('count step assign: {} and compare: {}'. format(assign, compare))
    fig = plt.figure()
    plt.plot(ls_assign, color = 'red', marker ='o')
    plt.plot(ls_compare, color = 'green', marker ='*')
    plt.legend(['assigns','compare'])
    plt.title('When k fixed')

def N_fixed(a, b, N = 20000):
    K = np.arange(10,1010,10)
    ls_compare = []
    ls_assign = []
    for k in K:
        A = np.random.randint(1,k,size=N)
        assign, compare = Bonus(A, a, b, k, N)
        ls_assign.append(assign)
        ls_compare.append(compare)
        print('When N fixed 20000, k : {}'.format(k))    
        print('count step assign: {} and compare: {}'. format(assign, compare))
    fig = plt.figure()
    plt.plot(ls_assign, color = 'red', marker ='o')
    plt.plot(ls_compare, color = 'green', marker ='*')
    plt.legend(['assigns','compare'])
    plt.title('When N fixed')
    

def Bai2():
    a = 1
    b = 10000
    # Case 1
    print('Case 1: k fixed')
    K_fixed(a, b, k = 100)

    print('*'*30)
    # Case 2
    print('Case 2: N fixed')
    N_fixed(a, b, N= 20000)

    plt.show()

# ----------------------- Main ----------------------------
if __name__ == '__main__':
    print('Bài 1: ')
    Bai1_1()

    print('*'*30)
    Bai1_2()

    print('*'*80)
    print('Bài 2: ')
    Bai2()

    print('END')

    