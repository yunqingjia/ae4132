'''
Spring 2021 AE 4132 HW5: rectangular elements for stress analysis
    NOTE: just using functions this time not building a class
author: yjia67
last updated: 04-15-2021
'''
import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import argparse
import cmd
import time

def read_input(filename: str):
    '''
    Parse input file
        * see PDF for input mesh format
    Args:
        - filename: name (and path) of the file to be parsed
    Returns:
        - nodes:    nodes dataframe [x_i, y_i, rx_i, ry_i, fx_i, fy_i]
        - els:      elements dataframe [n1_i, n2_i, n3_i, n4_i, E_i, nu_i, h_i]
    '''
    f = open(filename, 'r')

    # read nodes
    nnodes = int(f.readline())
    nodes = []
    for i in range(nnodes):
        node = list(map(float, f.readline().split()))
        nodes.append(node)

    # read elements
    nels = int(f.readline())
    els = []
    for i in range(nels):
        el = list(map(float, f.readline().split()))
        els.append(el)

    f.close()

    nodes_df = pd.DataFrame(nodes, columns=['x','y','rx','ry','fx','fy'])
    nodes_df[['rx', 'ry']] = nodes_df[['rx', 'ry']].astype(int)
    els_df = pd.DataFrame(els, columns=['n1','n2','n3','n4','E','nu','h'])
    els_df[['n1','n2','n3','n4']] = els_df[['n1','n2','n3','n4']].astype(int)

    print('Finished parsing.')
    print('%d nodes and %d elements.' % (nnodes, nels))

    return nodes_df, els_df

def compute_K(nodes: pd.DataFrame, els: pd.DataFrame):
    '''
    Construct the stiffness matrix
    Args:
        - nodes:    nodes dataframe
        - els:      elements dataframe
    Returns:
        - k:        global stiffness matrix
    '''
    # initialize K
    k_global = np.zeros((2*len(nodes), 2*len(nodes)))

    # use vectorization to compute the aspect ratio (gamma) 
    x1 = np.array(nodes.x[els.n1])
    x2 = np.array(nodes.x[els.n2])
    y2 = np.array(nodes.y[els.n2])
    y3 = np.array(nodes.y[els.n3])
    els['a'] = (x2-x1)
    els['b'] = (y3-y2)
    els['gamma'] = els.a/els.b

    # and the multiplication constant of each element
    els['const'] = els.h*els.E/(24*els.gamma*(1-els.nu)**2)
    g, v = els.gamma, els.nu
    p1 = (1+v)*g
    p2 = (1-3*v)*g
    p3 = 2+(1-v)*g**2
    p4 = 2*g**2+(1-v)
    p5 = (1-v)*g**2-4
    p6 = (1-v)*g**2-1
    p7 = 4*g**2-(1-v)
    p8 = g**2-(1-v)

    # i don't know an efficient way of constructing K 
    # from the above so just gonna use a loop
    for i in range(len(els)):
        n1, n2, n3, n4 = els.n1[i], els.n2[i], els.n3[i], els.n4[i]
        k_el = np.array([[ 4*p3[i],  3*p1[i],  2*p5[i], -3*p2[i], -2*p3[i], -3*p1[i], -4*p6[i],  3*p2[i]],
                         [ 3*p1[i],  4*p4[i],  3*p2[i],  4*p8[i], -3*p1[i], -2*p4[i], -3*p2[i], -2*p7[i]],
                         [ 2*p5[i],  3*p2[i],  4*p3[i], -3*p1[i], -4*p6[i], -3*p2[i], -2*p3[i],  3*p1[i]],
                         [-3*p2[i],  4*p8[i], -3*p1[i],  4*p4[i],  3*p2[i], -2*p7[i],  3*p1[i], -2*p4[i]],
                         [-2*p3[i], -3*p1[i], -4*p6[i],  3*p2[i],  4*p3[i],  3*p1[i],  2*p5[i], -3*p2[i]],
                         [-3*p1[i], -2*p4[i], -3*p2[i], -2*p7[i],  3*p1[i],  4*p4[i],  3*p2[i],  4*p8[i]],
                         [-4*p6[i], -3*p2[i], -2*p3[i],  3*p1[i],  2*p5[i],  3*p2[i],  4*p3[i], -3*p1[i]],
                         [ 3*p2[i], -2*p7[i],  3*p1[i], -2*p4[i], -3*p2[i],  4*p8[i], -3*p1[i],  4*p4[i]]
                        ])

        k_el = els.const[i]*k_el
        # assert np.allclose(k_el, k_el.T, atol = 1e-03) # check symmetry
        # again, don't think this is the most efficient way to do this but hope it works
        i1x, i1y, i2x, i2y, i3x, i3y, i4x, i4y = 2*n1, 2*n1+2, 2*n2, 2*n2+2, 2*n3, 2*n3+2, 2*n4, 2*n4+2
        k_global[i1x:i1y, i1x:i1y] += k_el[ :2,  :2]
        k_global[i1x:i1y, i2x:i2y] += k_el[ :2, 2:4]
        k_global[i1x:i1y, i3x:i3y] += k_el[ :2, 4:6]
        k_global[i1x:i1y, i4x:i4y] += k_el[ :2, 6: ]

        k_global[i2x:i2y, i1x:i1y] += k_el[2:4,  :2]
        k_global[i2x:i2y, i2x:i2y] += k_el[2:4, 2:4]
        k_global[i2x:i2y, i3x:i3y] += k_el[2:4, 4:6]
        k_global[i2x:i2y, i4x:i4y] += k_el[2:4, 6: ]

        k_global[i3x:i3y, i1x:i1y] += k_el[4:6,  :2]
        k_global[i3x:i3y, i2x:i2y] += k_el[4:6, 2:4]
        k_global[i3x:i3y, i3x:i3y] += k_el[4:6, 4:6]
        k_global[i3x:i3y, i4x:i4y] += k_el[4:6, 6: ]

        k_global[i4x:i4y, i1x:i1y] += k_el[6: ,  :2]
        k_global[i4x:i4y, i2x:i2y] += k_el[6: , 2:4]
        k_global[i4x:i4y, i3x:i3y] += k_el[6: , 4:6]
        k_global[i4x:i4y, i4x:i4y] += k_el[6: , 6: ]
        # assert np.allclose(k_global, k_global.T, atol = 1e-03) # check symmetry
        # print(k_global)

    return k_global

def apply_BC(nodes: pd.DataFrame, k_global: np.ndarray, f: np.ndarray):
    '''
    Apply displacement boundary conditions to both the stiffness matrix and the force vector
    Args:
        - nodes:    nodes dataframe
        - k_global: global stiffness matrix
        - f:        force vector
    Returns:
        - k_r:  reduced global stiffness matrix
        - f_r:  reduced force vector
        - idx:  indices where the displacement would be zero
    '''
    k_r = k_global
    f_r = f
    idx = []
    for i in range(len(nodes)):
        if nodes.rx[i] == 1:
            idx.append(2*i)
        if nodes.ry[i] == 1:
            idx.append(2*i+1)

    k_r = np.delete(k_r, obj=idx, axis=0)
    k_r = np.delete(k_r, obj=idx, axis=1)
    f_r = np.delete(f_r, idx)

    return k_r, f_r, idx

def compute_stress_strain(nodes: pd.DataFrame, els: pd.DataFrame):
    '''
    Compute the strain
    Args:
        - nodes:    nodes dataframe
        - els:      elements dataframe
    Returns:
        - nodes:    updated elements dataframe with stress and strain
    '''
    print(els)
    for i in range(len(els)):
        n1, n2, n3, n4 = els.n1[i], els.n2[i], els.n3[i], els.n4[i]
        ux1, ux2, ux3, ux4 = nodes.ux[[n1,n2,n3,n4]]


        # epsx = (ux2-ux1)/a + ((ux3-ux4)/(a*b) - (ux2-ux1)/(a*b))

def plot_results(nodes: pd.DataFrame, els: pd.DataFrame):
    '''
    Plot computed results
    # 1a) show deformed configuration
    # 1b) contour plots of all components of strain and stress over domain
    # 2a) max. nodal displacement over the domain
    # 2b) max. von Mises stress over the domain
    # 2c) strain energy of the entire beam
    '''


def solve_fea():
    '''
    The main function called to execute other methods
    '''
    ########## read input ########################
    fn = input('Input filename (without the .msh extension): ')
    nodes, els = read_input(fn + '.msh')

    ########## compute stiffness matrix ##########
    k = compute_K(nodes, els)

    ########## construct force vector ############
    f = np.zeros(2*len(nodes))
    f[::2] = nodes.fx
    f[1::2] = nodes.fy

    ########## apply boundary conditions #########
    k_r, f_r, idx = apply_BC(nodes, k, f)

    ########## compute displacement ##############
    u = np.linalg.inv(k_r) @ f_r
    # add back in the nodes where displacement = 0
    for i in idx:
        u = np.insert(u, i, 0)
    nodes['ux'] = u[::2]
    nodes['uy'] = u[1::2]
    nodes['xdef'] = nodes.x + nodes.ux
    nodes['ydef'] = nodes.y + nodes.uy

    ########## compute reaction forces ###########
    rxn = k @ u
    nodes['fx_rxn'] = rxn[::2]
    nodes['fy_rxn'] = rxn[1::2]
    print(nodes)

    ########## compute strain ####################
    els = compute_stress_strain(nodes, els)

    ########## compute stress ####################

    ########## plot results ######################
    # 1a) show deformed configuration
    # 1b) contour plots of all components of strain and stress over domain
    # 2a) max. nodal displacement over the domain
    # 2b) max. von Mises stress over the domain
    # 2c) strain energy of the entire beam    
    

if __name__ == '__main__':
    try:
        solve_fea()
    except FileNotFoundError:
        print('Error: file not found.')
    except ValueError:
        print('Error: input file format incorrect.')