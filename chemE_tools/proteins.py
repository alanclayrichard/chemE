import numpy as np
from matplotlib import pyplot as plt

def get_xyz(pdb_file,c_alpha=False):

    pdb1 = open(pdb_file)
    nlines = len(pdb1.readlines())
    x = np.zeros(nlines)
    y = np.zeros(nlines)
    z = np.zeros(nlines)
    j = 0
    with open(pdb_file) as pdb1:
        lines = pdb1.readlines()
        for i in range(nlines):
            if c_alpha == False:
                if lines[i][0:4]=='ATOM':
                    x[j] = float(lines[i][31:38])
                    y[j] = float(lines[i][39:46])
                    z[j] = float(lines[i][47:54])
                    j+=1
            else:
                if lines[i][0:4]=='ATOM' and lines[i][13:15]=='CA':
                    x[j] = float(lines[i][31:38])
                    y[j] = float(lines[i][39:46])
                    z[j] = float(lines[i][47:54])
                    j+=1
    x = np.trim_zeros(x,'b')
    y = np.trim_zeros(y,'b')
    z = np.trim_zeros(z,'b')
    return np.array([x,y,z]).T

def align_kabsch(r1,r2):
    # Align the proteins Using the Kabsch Algorithm
    centroid1 = np.array([np.mean(r1[:,0]),np.mean(r1[:,1]),np.mean(r1[:,2])])
    centroid2 = np.array([np.mean(r2[:,0]),np.mean(r2[:,1]),np.mean(r2[:,2])])
    r1 -= centroid1
    r2 -= centroid2
    covariance = np.dot(np.transpose(r1),r2)
    U,S,Vt = np.linalg.svd(covariance)
    R = np.dot(U,Vt)
    r1 = np.dot(r1,R)
    return r1,r2

def rmsd(xyz1,xyz2,align=False):
    if align==True:
        xyz1,xyz2 = align_kabsch(xyz1,xyz2)
    natoms1 = np.shape(xyz1)[0]
    natoms2 = np.shape(xyz2)[0]
    if natoms1 != natoms2:
        raise Exception("different number of atoms")
    else:
        rmsd = np.sqrt(np.sum((xyz2-xyz1)**2)/natoms1)
    return rmsd

def plot_pdb(pdb_file,c_alpha=False):
    if c_alpha==True:
        xyz = get_xyz(pdb_file,True)
    else:
        xyz = get_xyz(pdb_file)

    x_coordinate = xyz[:,0]
    y_coordinate = xyz[:,1]
    z_coordinate = xyz[:,2]
    n_atoms = len(xyz[:,0])

    c= np.arange(0,n_atoms,1)
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(x_coordinate,y_coordinate,z_coordinate,c=c, marker=".", cmap = 'viridis')