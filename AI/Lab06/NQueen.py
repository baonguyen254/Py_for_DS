# 18110053_NguyenQuocBao_AI_Lab6 

def display(solution_number, board):
    ''' Showing results through the position of placing Queen for each column'''
    row = '|   ' * len(board) + '|'
    hr  = '+---' * len(board) + '+'
    for col in board:
        print(hr)
        print(row[:col*4-3],'Q',row[col*4:])
    print(f'{hr}\n{board}\nSolution - {solution_number}\n')


def Issafe(q, board):
    ''' checking if placing a queen in a space would place it 
    in the same row, column, or diagonal as any other queens 
    '''
    x = len(board)
    for col in board:
        if col in [q+x,q-x]: return False
        x -= 1
    return True

def solve(boardsize, board=[], solutions_found=0):
    ''' A recursive utility function to solve N  
    Queen problem. 
    '''
    # If all queens are placed
    if len(board) == boardsize:
        solutions_found += 1
        display(solutions_found, board)
    else:
        # Consider this column and try placing this queen in all rows one by one
        for q in [col for col in range(1,boardsize+1) if col not in board]:
            if Issafe(q,board):
                solutions_found = solve(boardsize, board + [q], solutions_found)
    # If queen can not be place in any row in this column col then return board before
    return solutions_found

if __name__ == '__main__':
    solutions = solve(8)
    print(f'{solutions} solutions found')