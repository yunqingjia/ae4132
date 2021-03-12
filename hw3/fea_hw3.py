'''
Spring 2021 AE 4132 HW3
author: yjia67
last updated: 03-12-2021
'''
import numpy as np
import matplotlib.pyplot as plt

class FEAHW3():

    def __init__(self):
        pass

    def prob_1(self):

        ########## PROBLEM 1 ##########
        # define given parameters
        d = np.array([0.04, 0.02, 0.06]) # m
        A = np.square(d/2)*np.pi # m^2
        l = np.array([0.10, 0.10, 0.20]) # m
        E = 80e9 # Pa = N/m^2
        P = -3000 # N
        n = len(d)+1 # number of elements

        ### 1.1 Elementary Stiffnesss Matrices ###
        ke = E*A/l
        print('elementary stiffness coefficients (without pi)')
        print(ke/np.pi)

        ### 1.2 Global Stiffness Matrix and Force Vector ###
        k_global = np.zeros((n, n))
        for i in range(n-1):
            k_global[i:i+2, i:i+2] += ke[i]*np.array([[1, -1], [-1, 1]])

        F = np.array([0, 0, 0, -P]).reshape(4,1)

        print('global stiffness matrix (without pi)')
        print(k_global/np.pi)
        print('force vector')
        print(F)

        ### 1.3 Displacements at All Nodes ###
        # enforce displacement B.C.: u1 = 0
        # row 1 & col 1 of the global stiffness matrix go to 0
        # as well as the first element in the force vector
        k_global = k_global[1:, 1:]
        F = F[1:]
        u = np.linalg.inv(k_global) @ F
        u = np.insert(u, 0, 0)
        print('displacement at all nodes')
        print(u)

        ### 1.4 Plot the Dealio ###
        x = np.array([0.00, 0.10, 0.20, 0.40])
        plt.plot(x*100, u*100)
        # plt.ticklabel_format(axis='y', style='scientific')
        plt.xlabel('position x (cm)')
        plt.ylabel('displacement u (cm)')
        plt.title('HW3 P1 Displacement')
        plt.tight_layout()
        plt.savefig('hw3p1_displacement.jpg')
        plt.show()

        # compute dudx by differentiating displacement w.r.t. position
        dudx = np.insert(np.diff(u)/np.diff(x), 0, 0)
        plt.plot(x*100, dudx)
        plt.xlabel('position x (cm)')
        plt.ylabel(r'strain $\epsilon$')
        plt.title('HW3 P1 Strain')
        plt.tight_layout()
        plt.savefig('hw3p1_strain.jpg')
        plt.show()


        # compute stress by multiplying strain by E
        sig = dudx*E
        plt.plot(x*100, sig/10e9)
        plt.xlabel('position x (cm)')
        plt.ylabel(r'stress $\sigma$ (GPa)')
        plt.title('HW3 P1 Stress')
        plt.tight_layout()
        plt.savefig('hw3p1_stress.jpg')
        plt.show()

    def prob_2(self):

        ########## PROBLEM 2 ##########
        # define given parameters
        E = 70e9 # Pa
        A = 1e-5 # m^2
        l1, l2, l3 = 0.10, 0.06, 0.12
        l = np.array([l1, l2, l2, l1+l2, l3, l3])
        P = 3000 # N

        ### 2.1 FEM ###
        # 2.1.1 elementary stiffness
        k = E*A/l
        print('elementary stiffness coefficients (*10^-6)')
        print(k/1e6)

        ## 2.1.2 global stiffness matrix and force vector ##
        # #define starting and ending nodes lists and -1 bc index starts at 0
        # nodesi = [1, 2, 2, 1, 3, 3]-1
        # nodesj = [2, 3, 3, 3, 4, 4]-1
        # m = np.array([[1, -1], [-1, 1]])

        # jk gonna do this manually bc im too lazy to automate this
        # but ideally i could create an elementary matrix object/dict that stores the starting and ending node
        # after enforcing the B.C. @ u1, the global stiffness matrix looks like
        k1, k2, k3, k4, k5, k6 = k
        k_global = np.array([[ k1+k2+k3,           -k2-k3,        0],
                             [   -k2-k3,   k2+k3+k4+k5+k6,   -k5-k6], 
                             [        0,           -k5-k6,    k5+k6]])
        F = np.array([0, 0, P]).reshape(3,1)
        print('global stiffness matrix (*10^-6)')
        print(k_global/1e6)
        print('force vector')
        print(F)

        ## 2.1.3 displacements ##
        u = np.linalg.inv(k_global) @ F
        u = np.insert(u, 0, 0)
        print('displacement at all nodes')
        print(u)


        ## 2.2 Equivalent Springs ##
        k23 = self.kparallel(k2, k3)
        k56 = self.kparallel(k5, k6)
        k123 = self.kseries(k1, k23)
        k1234 = self.kparallel(k123, k4)
        keq = self.kseries(k1234, k56)

        print('equivalent spring stiffness (*10^-6')
        print(keq/1e6)

        u4eq = P/keq
        u3eq = P/k1234
        u2eq = (k2+k3)/(k1+k2+k3)*u3eq
        print('equivalent spring stiffness (*10^-6')
        print([0, u2eq, u3eq, u4eq])

        # # sanity check using hand derivaiton result
        # a = (k1*(k2+k3) + k4*(k1+k2+k3))/(k1+k2+k3)
        # b = k5+k6
        # keq1 = a*b/(a+b)
        # print(keq1/1e6)
    

    def kparallel(self, ki, kj):
        return ki+kj

    def kseries(self, ki, kj):
        return ki*kj/(ki+kj)



if __name__ == '__main__':

    hw3 = FEAHW3()

    print('PROBLEM 1 OUTPUT')
    hw3.prob_1()

    print('PROBLEM 2 OUTPUT')
    hw3.prob_2()