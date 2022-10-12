#  Phân tích thuật toán
#  Bài tập thực hành tuần 9
#  Nguyễn Quốc Bảo - 18110053

#=========================== Import ===========================

import numpy as np
from random import sample

#=========================== function Exercise 1===========================

def sum_of_squares_of_digits(value):
    return sum(int(c) ** 2 for c in str(value))


def find_the_smallest_cyclic(N):
    ls = [N]
    A = sum_of_squares_of_digits(N)
    while A not in ls:
        ls.append(A)
        A = sum_of_squares_of_digits(A)
    ls.append(A)

    return ls


def ex1():
    N = int(input('>> Input N: '))
    result = find_the_smallest_cyclic(N)
    print("smallest cyclic period for a given value of ",N)
    print(" -> ".join(str(x) for x in result))
    

#============================= main =============================

if __name__ == '__main__':
    print('\n')
    ex1()
    print('\n')