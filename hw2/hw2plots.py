'''
Spring 2021 AE 4132 HW2
author: yjia67
last updated: 02-22-2021
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

        x = np.linspace(0, L, n+1):
        y = [p1_u(xi) for xi in x]

    # Problem 2 Plotting: call on the functions
    def p2(self, x, N1, N2):

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

        if ((0 <= x) & (x <= 4*12)):
            return 90-x
        elif (x <= 8*12):
            return -6+x
        else:
            return

    # Problem 2 Case 2
    def p2_N2(self, x):

        if ((0 <= x) & (x <= 4*12)):
            return 24-x
        elif (x <= 8*12):
            return -72+x
        else:
            return

if __name__ == '__main__':

    hw2 = FEAHW2()

    ### PROBLEM 2 ###
    x1 = np.arange(0.0, 8.0*12, 0.01)
    hw2.p2(x1, hw2.p2_N1, hw2.p2_N2)