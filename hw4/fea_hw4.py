'''
Spring 2021 AE 4132 HW4: finite element program implementation from scratch
    NOTE:   very simplified version with a lot of assumptions:
            - 1-D bars in 2D plane
            - uniaxial stress
            - linearly elastic material
    Also note: not the most efficient implementation 
               could probably store nodes and elements as lists of dictionary objects instead
author: yjia67
last updated: 03-25-2021
'''
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import argparse
import cmd

class FEAHW4():

    def __init__(self):
        pass

    def read_input(self, filename: str):
        '''
        Parse the input text file into matrices

        Text file format:
            nnodes
            x_1 y_1 rx_1 ry_1 fx_1 fy_1
            x_2 y_2 rx_2 ry_2 fx_2 fy_2
            ...
            x_n y_n rx_n ry_n fx_n fy_n
            nels
            n1_1 n2_1 E_1 A_1
            n1_2 n2_2 E_2 A_2
            ...
            n1_n n2_n E_n A_n
        nnodes:         # of nodes
        (x_i, y_i):     coordinates of node i
        (rx_i, ry_i):   constraints in the x and y directions of node i. 1=constrainted, 0=free
        nels:           # of elements
        (n1_i, n2_i):   the first node and second element of element i

        Args:
            - filename: name (and path) of the file to be parsed
        Returns:
            - nodes:    (nnodes, 6) np array containing nodal info
            - els:      (nels, 4) np array containing elemental info
        '''

        f = open(filename, 'r')

        # read nodes
        nnodes = int(f.readline())
        nodes = np.zeros((nnodes, 6))
        for i in range(nnodes):
            nodes[i] = list(map(float, f.readline().split()))

        # read elements
        nels = int(f.readline())
        els = np.zeros((nels, 4))
        for i in range(nels):
            els[i] = list(map(float, f.readline().split()))

        f.close()

        print('Finished parsing.')
        print('%d nodes and %d elements.' % (nnodes, nels))

        return nodes, els

    def solve_fea(self, nodes: np.ndarray, els: np.ndarray):
        '''
        Put everything together

        Args:
            - nodes:    (nnodes, 6) np array containing nodal info
                        [x_i, y_i, rx_i, ry_i, fx_i, fy_i] for each node i
            - els:      (nels, 4) np array containing elemental info
                        [n1_i, n2_i, E_i, A_i] for each element i
        Returns:
            - u:        (nnodes, ) np array representing nodal displacement
            - rxn:      (nnodes, ) np array representing reaction forces
        '''
        # extract variables for easy access
        x, y, rx, ry, fx, fy = nodes.T

        # compute stiffness matrix
        k_global = self.compute_K(x, y, els)

        # construct force vector
        f = self.compute_F(fx, fy)

        # apply boundary conditions
        k_global_r, f_r, idx = self.apply_BC(rx, ry, k_global, f)

        # compute displacement
        u = np.linalg.inv(k_global_r) @ f_r
        # add back in the nodes where displacement = 0
        for i in idx:
            u = np.insert(u, i, 0)

        # solve for reaction forces
        rxn = k_global @ u

        return u, rxn

    def compute_K(self, x: np.ndarray, y: np.ndarray, els: np.ndarray):
        '''
        Construct the global stiffness matrix without boundary conditions

        Args:
            - x:    (nnodes, ) np array containing x coord for each node
            - y:    (nnodes, ) np array containing y coord for each node
            - els:  (nels, 4) np array containing elemental info
                    [n1_i, n2_i, E_i, A_i] for element i
        Returns:
            - k_global: (nnodes, nnodes) np array that represents
                        the global stiffness matrix as follows
                                    [u1_x, u1_y, ..., unnodes_x, unnodes_y]
                        [u1_x]
                        [u1_y]
                         ...
                        [unnodes_x]
                        [unnodes_y]
        '''

        # loop through each element to construct the global stiffness matrix
        k_global = np.zeros((2*len(nodes), 2*len(nodes)))
        for el in els:

            n1, n2, E, A = el
            # convert to int for indexing, which starts at 0 lol
            n1, n2 = int(n1-1), int(n2-1)

            # compute elemental stiffness
            x1, y1, x2, y2 = x[n1], y[n1], x[n2], y[n2]
            l = np.sqrt((x2-x1)**2 + (y2-y1)**2) # elemental length
            k= E*A/l
            theta = np.arctan2(y2-y1, x2-x1)
            R = self.compute_R(theta)
            k_el = k*R

            # assemble
            i11, i12, i21, i22 = 2*n1, 2*n1+2, 2*n2, 2*n2+2
            k_global[i11:i12, i11:i12] += k_el[:2, :2]
            k_global[i11:i12, i21:i22] += k_el[:2, -2:]
            k_global[i21:i22, i11:i12] += k_el[-2:, :2]
            k_global[i21:i22, i21:i22] += k_el[-2:, -2:]

        assert np.allclose(k_global, k_global.T, atol = 1e-03) # check symmetry

        return k_global

    def compute_F(self, fx: np.ndarray, fy: np.ndarray):
        '''
        Construct the force vector without boundary conditions

        Args:
            - fx:   (nnodes, ) np array containing force_x for each node
            - fy:   (nnodes, ) np array containing force_y for each node
        Returns:
            - f:    (2*nnodes, 1) np array that represents the force vector
                    [fx_1, fy_1, ..., fx_n, fy_n]
        '''
        f = np.zeros(2*len(fx))
        f[::2] = fx
        f[1::2] = fy
        return f

    def apply_BC(self, rx: np.ndarray, ry: np.ndarray, k_global: np.ndarray, f: np.ndarray):
        '''
        Apply displacement boundary conditions to both the stiffness matrix and the force vector

        Args:
            - nodes:    a size (nnodes, 6) numpy array containing nodal info
                        [x_i, y_i, rx_i, ry_i, fx_i, fy_i] for node i
            - k_global: the global stiffness matrix
            - f:        the force vector
        Returns:
            - k_r:  reduced global stiffness matrix with the B.C. applied
            - f_r:  reduced force vector with the B.C. applied
            - idx:  indices where the displacement would be zero
        '''

        idx = []

        for i in range(len(nodes)):
            if rx[i] == 1:
                idx.append(2*i)
            if ry[i] == 1:
                idx.append(2*i+1)

        k_global = np.delete(k_global, obj=idx, axis=0)
        k_global = np.delete(k_global, obj=idx, axis=1)
        f = np.delete(f, idx)

        k_global_r = k_global
        f_r = f
        
        return k_global_r, f_r, idx

    def compute_R(self, theta: float):
        '''
        Compute rotational matrix for a given theta

        Args:
            - theta: angle between the element coordinate and the global coordinate axes
        Returns:
            - R: the rotational matrix
        '''
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[ c**2,    c*s,  -c**2,   -c*s],
                        [  c*s,   s**2,   -c*s,  -s**2],
                        [-c**2,   -c*s,   c**2,    c*s],
                        [ -c*s,  -s**2,    c*s,   s**2]])
        return R

    def print_result(self, u: np.ndarray, rxn: np.ndarray):
        '''
        Print the nodal displacement and force vectors in easy-to-read manner

        Args:
            - u:    (nnodes, ) np array containing nodal displacements
            - rxn:  (nnodes, ) np array containing force information
        '''
        print('\t displacement \t force')
        print('\t u (m) \t\t f (N)')
        for i in range(len(u)):
            n = i+1
            if (n%2 == 1):
                print('node%dx:\t %0.5f \t %0.2f' % ((n+1)//2, u[i], rxn[i]))
            else:
                print('node%dy:\t %0.5f \t %0.2f' % (n//2, u[i], rxn[i]))

    def plot_results(self, u: np.ndarray, rxn: np.ndarray, nodes: np.ndarray, els: np.ndarray, scl: int):
        '''
        Plot the original structure and the deformed structure

        Args:
            - u:    (nnodes, ) np array containing nodal displacements
            - rxn:  (nnodes, ) np array containing force information
            - nodes:    (nnodes, 6) np array containing nodal info
                        [x_i, y_i, rx_i, ry_i, fx_i, fy_i] for each node i
            - els:      (nels, 4) np array containing elemental info
                        [n1_i, n2_i, E_i, A_i] for each element i
        '''
        x, y, _, _, _, _ = nodes.T
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) # in case i want multiple plots

        ax1.set_aspect('equal')
        x_def, y_def = x + u[::2], y +  u[1::2]
        x_plt, y_plt = x + scl*u[::2], y + scl*u[1::2]
        stress = []

        for el in els:
            n1, n2, E, A = el
            n12 = [int(n1-1), int(n2-1)]

            # compute stress in each bar
            [x1, x2], [y1, y2] = x[n12], y[n12]
            [xd1, xd2], [yd1, yd2] = x_def[n12], y_def[n12]
            [xplt1, xplt2], [yplt1, yplt2] = x_plt[n12], y_plt[n12]

            l = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            ld = np.sqrt((xd2-xd1)**2 + (yd2-yd1)**2)
            ld_plt = np.sqrt((xplt2-xplt1)**2 + (yplt2-yplt1)**2)
            stress.append(E*ld_plt)

            # determine compression (shortening) vs tension (elongation)
            ax1.plot(x[n12], y[n12], color='gray')
            if (ld == l): # net force zero
                ax1.plot(x_plt[n12], y_plt[n12], color='k')
            elif (ld > l): # tension
                ax1.plot(x_plt[n12], y_plt[n12], color='r')
            elif (ld < l): # compression
                ax1.plot(x_plt[n12], y_plt[n12], color='b')

        ax1.set_xlabel(r'$x$ (m)')
        ax1.set_ylabel(r'$y$ (m)')
        ax1.set_title('Unloaded vs Loaded (red=tension, blue=compression)\nDisplacement Scale=%d' % scl)

        # not the most efficicent implementation but o well

        ax2.set_aspect('equal')
        s_max, s_min = max(stress), min(stress)
        cm = plt.get_cmap('jet')
        stress = np.array(stress).reshape(len(stress),1)
        els_stress = np.hstack((els, stress))
        for el in els_stress:
            n1, n2, E, A, s = el
            n12 = [int(n1-1), int(n2-1)]
            f = (s-s_min) / (s_max-s_min)
            color = cm(f)
            ax2.plot(x_plt[n12], y_plt[n12], '-', color=color)

        ax2.set_xlabel(r'$x$ (m)')
        ax2.set_ylabel(r'$y$ (m)')
        ax2.set_title('Element Stresses in Deformed Configuration\nDisplacement Scale=%d' %  scl)
        plt.tight_layout()

        # add the color bar
        s_scale = 1e-9
        norm = colors.Normalize(vmin=s_scale*s_min, vmax=s_scale*s_max)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([s_min, s_max])
        cb = plt.colorbar(sm)
        cb.set_label('Stress (GPa)')

        figname = input('Save the deformed structure plot as (without the .jpg extension): ')
        

        plt.savefig(figname+'.jpg')

        plt.show()

if __name__ == '__main__':

    hw4 = FEAHW4()

    filename = input('Enter input filename (without the .txt extension): ')

    try:
        nodes, els = hw4.read_input(filename + '.txt')
        u, rxn = hw4.solve_fea(nodes, els)

        disp_scale = 10 # scale the displacement for better visualization
        hw4.print_result(u, rxn)
        hw4.plot_results(u, rxn, nodes, els, disp_scale)

    except FileNotFoundError:
        print('Error: file not found.')
    except ValueError:
        print('Error: input file format incorrect.')