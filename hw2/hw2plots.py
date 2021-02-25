'''
Spring 2021 AE 4132 HW2
author: yjia67
last updated: 02-24-2021
'''
import numpy as np
import matplotlib.pyplot as plt

class FEAHW2():

    def __init__(self):
        pass

    # Problem 1.4
    def p1_4(self, x):
        # Define the input parameters for the specific case
        P = 400 # N
        q = 100 # N/m
        L = 2 # m
        A = 0.0003 # m^2
        E = 70e9 # Pa
        v = 0.3
        n = 50 # elements
        le = L/n # length of each element
        ke = E*A/le

        u = self.p1_compute_u(ke, n)

        #######################################################
        # I was able to compute all the coefficients 
        # but ran out of time trying to figure out how to plot
        #######################################################

        # Plot the piecewise functions
        # x_lst = np.arange()
        # y_lst = np.array([])
        # for i in range(n-1):
        #     x = np.arange(i*le, (i+1)*le, 1)
        #     print(x)
        #     y = np.array([self.p1_uhat(xi, u[i], u[i+1], i, le) for xi in x])
        #     x_lst = np.append(x_lst, x)
        #     y_lst = np.append(y_lst, y)
        # print(x_lst.shape)

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

    # Define the piecewise displacement function
    def p1_uhat(self, x, u1, u2, i, le):
        uhat = u1 + (u2-u1)/le*(x-i*le)
        return uhat

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

    # Problem 3
    def p3(self, b, P1, P2):

        plt.plot(b, list(map(P1, b)), color='k', ls='-', label='1: quad. RR')
        plt.plot(b, list(map(P2, b)), color='b', ls='-', label='2: trig. RR')
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$\Pi(x)$')
        plt.legend()
        plt.title('HW2 Problem 3')
        plt.savefig('hw2p3.png')
        plt.show()


    def p3_pi1(self, beta):
        return -5/216*np.square(1+3*beta)

    def p3_pi2(self, beta):
        pi = np.pi
        return -80*np.square(-2+pi+pi*beta)/(3*pi**6)

if __name__ == '__main__':

    hw2 = FEAHW2()

    ### PROBLEM 1 ###
    x1 = np.arange(0.0, 2.0, 0.01)
    hw2.p1_4(x1)

    ### PROBLEM 2 ###
    x2 = np.arange(0.0, 8.0*12, 0.01)
    hw2.p2(x2, hw2.p2_N1, hw2.p2_N2)

    ### PROBLEM 3 ###
    beta = np.arange(-10, 10, 0.01)
    hw2.p3(beta, hw2.p3_pi1, hw2.p3_pi2)
