import numpy as np
import time

def check(grid, num, x, y):
    for i in range(9):
        if grid[i][y] == num:
            return False
    for j in range(9):
        if grid[x][j] == num:
            return False
    x0 = (x//3) * 3
    y0 = (y//3) * 3
    for i in range(3):
        for j in range(3):
            if grid[x0+i][y0+j] == num:
                return False
    return True

def solve(grid):
    start = time.time()
    for i in range(9 + 1):
        if i==9:
            return True, "Solved in %.4fs" % (time.time() - start)
        for j in range(9):
            if grid[i][j] == 0:
                for num in range(1,10):
                    if check(grid, num, i, j):
                        grid[i][j] = num
                        if solve(grid)[0]:
                            return True, "Solved in %.4fs" % (time.time() - start)
                        grid[i][j] = 0
                return False, "Solved in %.4fs" % (time.time() - start)

if __name__ == "__main__":
    grid = np.array([[5,3,0,0,7,0,0,0,0],
                    [6,0,0,1,9,5,0,0,0],
                    [0,9,8,0,0,0,0,6,0],
                    [8,0,0,0,6,0,0,0,3],
                    [4,0,0,8,0,3,0,0,1],
                    [7,0,0,0,2,0,0,0,6],
                    [0,6,0,0,0,0,2,8,0],
                    [0,0,0,4,1,9,0,0,5],
                    [0,0,0,0,8,0,0,7,9]], dtype=np.uint8)

    solution, time = solve(grid)
    print(solution)
    print(time)

