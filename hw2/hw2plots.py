'''
Spring 2021 AE 4132 HW2
author: yjia67
last updated: 02-24-2021
'''
import numpy as np
import matplotlib.pyplot as plt

# why did i write this in an abstrac / object-oriented manner lol
# don't / won't do it again

class FEAHW2():

    def __init__(self):
        pass

    # Problem 1.4
    def p1_4(self):
        # Define the input parameters for the specific case
        P = 400 # N
        q = 100 # N/m
        L = 2 # m
        A = 0.0003 # m^2
        E = 70e9 # Pa
        v = 0.3
        n = 50 # elements

        # Define the spring constant (same for all elements)
        le = L/n
        ke = E*A

        u = self.p1_compute_u(ke, n)

        # Plot the piecewise functions


    def p1_compute_u(self, ke, n):
        '''
        Compute the u_i values from the matrix that represent the system of equations
        of the partial derivatives of Pi_hat

        The coefficient matrix is formatted as the following:
        [[ 0,  0,  0,  0, ... ,  0,  0],
         [ 0,  2, -1,  0, ... ,  0,  0],
         [ 0, -1,  2, -1, ... ,  0,  0],
                ... ...
         [ 0,  0,  0,  0, ... , -1,  1]]


        input:  ke = spring constant
                n = # of elements
        output: u = 1D np.Array of coefficients in linear approximation

        '''
        # Initialize an NxN matrix where N = # of elements
        mat = np.zeros((n, n))

        # Loop from 1:n and populate the matrix and solve for constants (u_i)
        # Since we know u1 = 0 from the B.C., the first column will remain 0
        for i in range(1, n-1):
            mat[i][i-1:i+2] = [-1, 2, -1]

        # Manually set the first column of the second row to 0
        # Manually set the last row of the matrix to [..., -1, 1]
        # Multiply everything by ke & remove the first row and first column (all 0s)
        mat[1][0] = 0
        mat[-1][-2:] = [-1, 1]
       
        mat = ke*mat[1:, 1:]
        u, v = np.linalg.eig(mat)
        
        # Insert u1=0 at the start of the array
        u = np.insert(u, 0, 0.0)

        return u

    # Problem 2 Plotting: call on the functions
    def p2(self, x, N1, N2):
        '''
        Plotting everything on the same plot:
        Case 1: Quadratic Rayleigh-Ritz
                Governing Equation
                Linear Rayleigh-Ritz
        Case 2: same dealio
        '''

        plt.plot(x, list(map(N1, x)), color='k', ls='-', label='1: quad. RR')
        plt.plot(x, list(map(N1, x)), color='r', ls='--', label='1: gov. eqn')
        plt.plot(x, np.ones(len(x))*66, color='m', label='1: lin. RR')
        plt.plot(x, list(map(N2, x)), color='b', ls='-', label='2: quad. RR')
        plt.plot(x, list(map(N2, x)), color='y', ls='--', label='2: gov. eqn')
        plt.plot(x, np.ones(len(x))*24, color='c', label='2: lin. RR')
        plt.xlabel(r'$x$ $(in)$')
        plt.ylabel(r'$N(x)$ $(lb_f)$')
        plt.xticks(np.arange(min(x), max(x)+1, 6))
        plt.legend()
        plt.title('HW2 Problem 2')
        plt.savefig('hw2p2.png')
        plt.show()

    # Problem 2 Case 1
    def p2_N1(self, x):

        if (x <= 4*12):
            return 90-x
        else:
            return -6+x

    # Problem 2 Case 2
    def p2_N2(self, x):

        if (x <= 4*12):
            return 24-x
        else:
            return -72+x

if __name__ == '__main__':

    hw2 = FEAHW2()

    ### PROBLEM 1 ###
    hw2.p1_4()

    ### PROBLEM 2 ###
    # define range for x
    # x1 = np.arange(0.0, 8.0*12, 0.01)
    # hw2.p2(x1, hw2.p2_N1, hw2.p2_N2)

    ### PROBLEM 3 ###
