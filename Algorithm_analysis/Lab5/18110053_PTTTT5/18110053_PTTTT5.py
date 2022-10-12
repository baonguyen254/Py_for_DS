#  Phân tích thuật toán
# Bài tập thực hành tuần 5
#  Nguyễn Quốc Bảo - 18110053

#------------------------------------------------
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import seaborn as sns

np.seterr(all='warn')

#------------------------------------------------


def multiply_Matrix(_matrix_A,_matrix_B):
    global counter_assign
    global counter_compare

    # initialization matrix C contain result matrix A multiply matrix B
    C = np.zeros_like(_matrix_A) 
    for i in range(len(_matrix_A)):
        counter_compare += 1
        for j in range(len(_matrix_A)):
            counter_compare += 1
            for k in range(len(_matrix_A)):
                C[i,j] += (_matrix_A[i,k]*_matrix_B[k,j])
                counter_assign += 1
                counter_compare += 1
    return C



def strassen(_matrix_A,_matrix_B):
    """
        Implementation of the strassen algorithm.
    """
    global counter_assign
    global counter_compare
    
    
    counter_compare += 1
    if len(_matrix_A) == 1:
        return _matrix_A*_matrix_B # conventionally computed
    else:
        n_new = len(_matrix_A) // 2
        counter_assign += 1
        
        # dividing the matrices in 4 sub-matrices:
        # matrix A into a11 , a12, a21, a22.
        # matrix B into b11 , b12, b21, b22.

        # Calculating p1 to p7:

        # p1 = (a11+a22) * (b11+b22)
        p1 = strassen(_matrix_A[:n_new, :n_new] + _matrix_A[n_new:,n_new:], _matrix_B[:n_new,:n_new] + _matrix_B[n_new:,n_new:])  

        # p2 = (a21+a22) * (b11)
        p2 = strassen(_matrix_A[n_new:,:n_new] + _matrix_A[n_new:,n_new:], _matrix_B[:n_new, :n_new])  

        # p3 = (a11) * (b12 - b22)
        p3 = strassen(_matrix_A[:n_new, :n_new],_matrix_B[:n_new, n_new:] - _matrix_B[n_new:,n_new:])  

        # p4 = (a22) * (b21 - b11)
        p4 = strassen(_matrix_A[n_new:,n_new:], _matrix_B[n_new:,:n_new] - _matrix_B[:n_new, :n_new] )  

        # p5 = (a11+a12) * (b22)
        p5 = strassen(_matrix_A[:n_new, :n_new] + _matrix_A[:n_new, n_new:], _matrix_B[n_new:,n_new:])  

        # p6 = (a21-a11) * (b11+b12)
        p6 = strassen(_matrix_A[n_new:,:n_new] + _matrix_A[:n_new, :n_new] , _matrix_B[:n_new, :n_new] + _matrix_B[:n_new, n_new:])  

        # p7 = (a12-a22) * (b21+b22)
        p7 = strassen( _matrix_A[:n_new, n_new:] - _matrix_A[n_new:,n_new:], _matrix_B[n_new:,:n_new] + _matrix_B[n_new:,n_new:])  

        counter_assign += 7

        # calculating c11, c12, c21 and c22:

        c11 = p1 + p4 - p5 + p7  # c11 = p1 + p4 - p5 + p7
        
        c12 = p3 + p5 # c12 = p3 + p5

        c21 = p2 + p4 # c21 = p2 + p4

        c22 = p1 + p3 - p2 + p6  # c22 = p1 + p3 - p2 + p6
        
        counter_assign += 4

        
        # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
        C = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22)))) 
        counter_assign += 1

        return C


def plot_compare_complexity(N, n_compare, n_assign, tille = None):
    N_log7 = [n**np.log2(7) for n in N]
    N_3 = [n**3 for n in N]
    fig = plt.figure()
    sns.lineplot(N, N_3)
    sns.lineplot(N, N_log7)
    sns.lineplot(N, n_assign)
    sns.lineplot(N, n_compare)
    plt.legend(['N^3','N^log(7)','n_assign','n_compare'])
    plt.title('plot complexity '+ tille)
    plt.xlabel('N')
    plt.show()
    del N_3, N_log7, n_assign, n_compare
    tille = tille.replace(' ','_') 
    plt.savefig(tille+'.png')

def ex1():
    global counter_assign
    global counter_compare
    print("-"*40,'Exercise 1','-'*40)
    K = np.arange(2,10,1)
    N = np.array([2**k for k in K], dtype= np.int64)
   
    ls_n_compare1, ls_n_compare2  = [],[]
    ls_n_assign1, ls_n_assign2 = [],[]


    for n in N:
        print('-'*50)
        print('>> N = ',n)
        A = np.random.randint(1,1001,size=(n,n))
        B = np.random.randint(1,1001,size=(n,n))
        print('>> A = \n',A)
        print('>> B = \n',B)
        counter_assign = 0
        counter_compare = 0
        C = multiply_Matrix(A,B)
        print('----------------By Traditional medthod----------------')
        print('C = A.B = ')
        print(np.matrix(C))
        print("When N = {} , then count_compare: {} and count_assign: {}".format(n, counter_compare,counter_assign))

        ls_n_assign1.append(counter_assign)
        ls_n_compare1.append(counter_compare)
    
        del C
        print('----------------By Strassen medthod----------------')
        counter_assign = 0
        counter_compare = 0
        C = strassen(A,B)
        print('C = A.B = ')
        print(np.matrix(data = C))
        print("When N = {} , then count_compare: {} and count_assign: {}".format(n, counter_compare,counter_assign))

        ls_n_assign2.append(counter_assign)
        ls_n_compare2.append(counter_compare)


    plot_compare_complexity(N,ls_n_compare1,ls_n_assign1,tille='Naive algorithm')
    plot_compare_complexity(N,ls_n_compare2,ls_n_assign2,tille='Strassen algorithm')


if __name__ == "__main__":
    ex1()