# Nguyễn Quốc Bảo - 18110053
# Lab3 - PTTT 
# ---------------------------------------------------------
from math import *
import numpy as np
import queue

from numpy.core.defchararray import join

counter_assign = 0
counter_compare = 0

# ---------------------------------------------- Exercise 1 ----------------------------------------------------
def fix_valid(equation):
    equation = equation.replace(" ", "")
    equation = equation.replace("^", "**")
    equation = equation.replace("-","+ (-1)*")
    return equation

def eval_polynomial(equation, x_value):
    fixed_equation = fix_valid(equation.strip())
    parts = fixed_equation.split("+")
    x_str_value = str(x_value)
    parts_with_values = (part.replace("n", x_str_value) for part in parts )
    partial_values = (eval(part) for part in parts_with_values)
    return sum(partial_values)

def compute_exponent(equation,N):
    temp = 0
    for n in N:
        F = eval_polynomial(equation,n)
        alpha = log(abs(F))/log(n)
        M = max(alpha, temp)
        temp = alpha
    return int(M)

def find_maximum_degree(equation):
    fixed_equation = fix_valid(equation.strip())
    parts = fixed_equation.split("+")
    degree = []
    for part in parts:
        idx = part.find('**')
        while (part[idx].isdigit() != True):
            if (part[idx] == 'n' ):
                return 'n'
            idx += 1
        degree.append(part[idx])
    return max(degree)

def demo_ex1():
    a = 10;b = 1000
    N = np.arange(a,b+1,1)
    f_n = ["n^2", "n^3 + cos(n)*n^4", "n^n","n^3 + n^2 + n + 1"]
    print('a = {} and b = {}'.format(a,b))
    for i,f in zip(range(1,len(f_n)+1),f_n):
        alpha = compute_exponent(f, N)
        degree_f = find_maximum_degree(f)
        if (degree_f != 'n'):
            degree_f = int(degree_f)
        else:
            degree_f = b
        print(i,')', 'f(n) = ',f)
        if (alpha == degree_f):
            print('when f(n) = {} then f(n) = O(n^{})'.format(f,alpha))
        else:
            print("f(n) has no form O(n^{})".format(alpha))


def ex1():
    print("check with the following cases: ")
    demo_ex1()
    print()
    f = input('>> input f(n): ')
    a = int(input('>> Input number a: '))
    b = int(input('>> Input number b: '))
    
    N = np.arange(a,b+1,1)

    alpha = compute_exponent(f, N)
    degree_f = find_maximum_degree(f)
    if (degree_f != 'n'):
        degree_f = int(degree_f)
    else:
        degree_f = b
    if (alpha == degree_f):
        print('when f(n) = {} then f(n) = O(n^{})'.format(f,alpha))
    else:
        print("f(n) has no form O(n^{})".format(alpha))


# ---------------------------------------------- Exercise 2 ----------------------------------------------------


# Phương pháp cổ điển
def multiply(A,B):
    global counter_assign
    global counter_compare
    #--- Nhân từng phần tử từ phải sang trái ---
    result = []
    for i in range(len(B)-1,-1,-1):
        memory = 0
        ls = [0]*(len(B)-i-1)
        counter_assign += 2
        for j in range(len(A)-1,-1,-1):
            r = int(B[i])*int(A[j])
            if (memory != 0):
                r = r + memory
                memory = 0
            temp = r
            if (temp >= 10):
                k = temp%10
                memory = int(temp/10)
            else:
                k = r
            ls.append(k)
            counter_compare += 2
            counter_assign += 6


        while (len(ls) != (len(A)+len(B))):
            ls.append(0)
        ls.reverse()
        result.append(ls)
        # print(ls)
    result = np.array(result)


    #---Cộng các giá trị vừa nhân theo hàng dọc---
    Sum = list(sum(result))
    memory = 0
    counter_assign += 2
    for i in range(len(Sum)-1,-1,-1):
        temp = Sum[i]
        if (memory != 0):
            temp = temp + memory
            memory = 0
        if (temp >= 10):
            k = temp%10
            memory = int(temp/10)
        else:
            k = temp
        Sum[i] = k
        counter_compare += 2
        counter_assign += 4
    while (Sum[0] == 0):
        Sum.pop(0)

    return Sum


def karatsuba(X, Y):
    global counter_assign
    global counter_compare

    counter_compare += 1
    # -- Base case --
    if X < 10 and Y < 10:
        return X * Y
    

    # --- determine the size of X and Y ---
    size = max(len(str(X)), len(str(Y)))

    # --- Split X and Y ---
    n = ceil(size/2)
    p = 10 ** n
    a = floor(X // p)
    b = X % p
    c = floor(Y // p)
    d = Y % p
    counter_assign += 7

    # --- Recur until base case ---
    ac = karatsuba(a, c)
    bd = karatsuba(b, d)
    e = karatsuba(a + b, c + d) - ac - bd

    # --- return the equation ---
    return int(10 ** (2 * n) * ac + (10 ** n) * e + bd)

def ex2():
    global counter_assign
    global counter_compare
    A = input('>> input number A: ')
    B = input('>> input number B: ')
    print("-------------By classic method-------------")
    counter_assign = 0
    counter_compare = 0
    C = multiply(A,B)

    print(">> C = A.B =",''.join([str(item) for item in C ]))
    print("count assign: {} step and count compare: {} step".format(counter_assign,counter_compare))
    print('-'*80)

    print("-------------By Karatsuba method-------------")
    counter_assign = 0
    counter_compare = 0
    K = karatsuba(int(A),int(B))
    print(">> C = A.B =", K)
    print("count assign: {} step and count compare: {} step".format(counter_assign,counter_compare))
    
    # Check the cases following:
    print("--------------- Check cases n = 2^k with k= 10, 11,...32 -----------------")
    K = np.arange(10,32)
    N = [2**k for k in K]
    for n in N:
        A = np.random.randint(9,size = n)
        B = np.random.randint(9,size = n)
        A = ''.join(map(str,A))
        B = ''.join(map(str,B))
        print("-"*100)
        print(">> N = {} ".format(n))
        print(">> A = ",A) 
        print(">> B = ",B)
        print("-------------By classic method-------------")
        counter_assign = 0
        counter_compare = 0
        C = multiply(A,B)

        print(">> C = A.B =",''.join([str(item) for item in C ]))
        print("When N = {}, count assign: {} step and count compare: {} step".format(n, counter_assign,counter_compare))
        print('-'*80)

        print("-------------By Karatsuba method-------------")
        counter_assign = 0
        counter_compare = 0
        K = karatsuba(int(A),int(B))
        print(">> C = A.B =", K)
        print("When N = {},  count assign: {} step and count compare: {} step".format(n,counter_assign,counter_compare))


if __name__ == '__main__':
    print("-----------Exercise 1------------")
    ex1()
    print('\t\t END EX1 \t\t')
    print("-----------Exercise 2------------")
    ex2()
    print('\t\t END EX2 \t\t')
