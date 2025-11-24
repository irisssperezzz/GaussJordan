import numpy as np


'''
The algorithm is originally and credited to "Murad Elarbi"
Others such as gui and implementation of user inputs are made
by Iris
'''

def gaussjordan(a, b):
    a = np.array(a, float)
    b = np.array(b, float)
    length = len(b)

    # main loop from the algorithm
    for k in range(length):
        # partial pivoting
        if np.fabs(a[k, k]) < 1.0e-12:
            for i in range(k + 1, length):
                if np.fabs(a[i, k]) > np.fabs(a[k, k]):
                    for j in range(k, length):
                        a[k, j], a[i, j] = a[i, j], a[k, j]
                    b[k], b[i] = b[i], b[k]
                    break
        # division of the pivot row
        pivot = a[k, k]
        for j in range(k, length):
            a[k, j] /= pivot
        b[k] /= pivot
        # elimination loop
        for i in range(length):
            if i == k or a[i, k] == 0: continue
            factor = a[i, k]
            for j in range(k, length):
                a[i, j] -= factor * a[k, j]
            b[i] -= factor * b[k]
    return b, a

def main():
    # sample TODO: remove later and implement unit testing
    a = [[0, 2, 0, 1],
         [2, 2, 3, 2],
         [4, -3, 0, 1],
         [6, 1, -6, -5]]
    b = [0, -2, -7, 6]

    X, A = gaussjordan(a, b)

    print(f"Solution \n {X}")
    print(f"Transformed Matrix[A] \n {A}")

# TODO: implement user input
if __name__ == "__main__":
    main()
