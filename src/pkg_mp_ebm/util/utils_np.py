import numpy as np
from util.datatype import *

from typing import List

### Value functions
def np_sigmoid(x):
    return 1/(1 + np.exp(-x))

def np_gaussian(xy:tuple, mu:tuple, sigma:tuple, rho):
    in_exp = -1/(2*(1-rho**2)) * ((xy[0]-mu[0])**2/(sigma[0]**2) 
                                + (xy[1]-mu[1])**2/(sigma[1]**2) 
                                - 2*rho*(xy[0]-mu[0])/(sigma[0])*(xy[1]-mu[1])/(sigma[1]))
    z = 1/(2*np.pi*sigma[0]*sigma[1]*np.sqrt(1-rho**2)) * np.exp(in_exp)
    return z

### Create masks
def np_create_circle(r:int, ring:bool=False):
    # Generate a n-by-n matrix, n=2r+1
    # This matrix contains a circle filled with 1, otherwise 0
    # If 'ring' is true, only the edge of the circle is 1.
    A = np.arange(-r,r+1)**2
    dists = np.sqrt(A[:,None] + A)
    if ring:
        return (np.abs(dists-r)<0.5).astype(int)
    else:
        return (dists<r).astype(int)

def np_create_circle_mask(circle_centre:CoordsCartesian, r:int, base_matrix:NumpyImageSC):
    # Create a circle mask on an empty matrix with the given shape
    # If the maskis out of borders, it will be cut.
    base_matrix = np.zeros(base_matrix.shape)
    np_circle = np_create_circle(r)
    row_min = np.maximum(circle_centre[1]-r, 0)
    row_max = np.minimum(circle_centre[1]+r, base_matrix.shape[0]-1)
    col_min = np.maximum(circle_centre[0]-r, 0)
    col_max = np.minimum(circle_centre[0]+r, base_matrix.shape[1]-1)
    if row_max == base_matrix.shape[0]-1:
        if -(r-(base_matrix.shape[0]-1-circle_centre[1])) != 0:
            np_circle = np_circle[:-(r-(base_matrix.shape[0]-1-circle_centre[1])),:]
    if row_min == 0:
        np_circle = np_circle[r-circle_centre[1]:,:]
    if col_max == base_matrix.shape[1]-1:
        if -(r-(base_matrix.shape[1]-1-circle_centre[0])) != 0:
            np_circle = np_circle[:, :-(r-(base_matrix.shape[1]-1-circle_centre[0]))]
    if col_min == 0:
        np_circle = np_circle[:, r-circle_centre[0]:]
    base_matrix[row_min:row_max+1, col_min:col_max+1] = np_circle
    return base_matrix

### Matrix operations
def np_matrix_subtract(base_matrix:NumpyImageSC, ref_matrix:NumpyImageSC):
    # Subtract the intersection area of the base with the reference 
    intersection = base_matrix.astype(int) & ref_matrix.astype(int)
    base_matrix -= intersection
    return base_matrix

### Location mask map
def np_dist_map(centre:CoordsCartesian, base_matrix:NumpyImageSC):
    # Create a distance map given the centre and map size
    # The distance is defined as the Euclidean distance
    # The map is normalized
    base_matrix = np.zeros(base_matrix.shape)
    x = np.arange(0, base_matrix.shape[1])
    y = np.arange(0, base_matrix.shape[0])
    x, y = np.meshgrid(x, y)
    base_matrix = np.linalg.norm(np.stack((x-centre[0], y-centre[1])), axis=0)
    return base_matrix/base_matrix.max()

def np_gaudist_map(centre:CoordsCartesian, base_matrix:NumpyImageSC, sigmas:Indexable=[100,100], rho:float=0, normal:bool=False):
    # Create a distance map given the centre and map size
    # The distance is defined by the Gaussian distribution
    # The map is normalized
    sigma_x, sigma_y = sigmas[0], sigmas[1]
    x = np.arange(0, base_matrix.shape[1])
    y = np.arange(0, base_matrix.shape[0])
    x, y = np.meshgrid(x, y, indexing='xy')
    in_exp = -1/(2*(1-rho**2)) * ((x-centre[0])**2/(sigma_x**2) 
                                + (y-centre[1])**2/(sigma_y**2) 
                                - 2*rho*(x-centre[0])/(sigma_x)*(y-centre[1])/(sigma_y))
    z = 1/(2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2)) * np.exp(in_exp)
    if normal:
        return z/z.max()
    return z

def get_patch(traj:np.ndarray, height:int, width:int, base_matrix:NumpyImageSC) -> List[NumpyImageSC]:
    H, W = height, width
    x = np.round(traj[:,0]).astype('int')
    y = np.round(traj[:,1]).astype('int')
    x_low = base_matrix.shape[1] // 2 - x
    x_up = base_matrix.shape[1] // 2 + W - x
    y_low = base_matrix.shape[0] // 2 - y
    y_up = base_matrix.shape[0] // 2 + H - y
    patch = [base_matrix[y_l:y_u, x_l:x_u] for x_l, x_u, y_l, y_u in zip(x_low, x_up, y_low, y_up)]
    return patch

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    centre1 = (30,40)
    centre2 = (60,60)
    centre3 = (80,80)
    centres = [centre1, centre2, centre3]

    X = np.arange(0, 120)
    Y = np.arange(0, 100)
    X, Y = np.meshgrid(X, Y)
    base = np.zeros((100,120))

    map_d = np_dist_map(centre1, base)

    map1 = np_gaudist_map(centre1, base, sigmas=[10,10], rho=0, flip=False)
    map2 = np_gaudist_map(centre2, base, sigmas=[10,10], rho=0, flip=False)
    map3 = np_gaudist_map(centre3, base, sigmas=[10,10], rho=0, flip=False)

    map = map1 + map2/2 + map3/3
    map = map/np.max(map)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, map, cmap='hot')

    plt.imshow(map1)
    [plt.plot(c[0],c[1],'rx') for c in centres]

    plt.show()