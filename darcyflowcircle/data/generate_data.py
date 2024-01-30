import ufl
from dolfinx import fem, io, mesh, plot,default_scalar_type
import gmsh
from mpi4py import MPI
import meshio
from tqdm import trange
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import meshio
from scipy.spatial import Delaunay
import numpy as np
from scipy.fftpack import idct
from scipy import interpolate
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import csr_matrix,save_npz
import scipy.sparse
gmsh.initialize()
membrane = gmsh.model.occ.addDisk(0.5, 0.5, 0, 0.45, 0.45)
gmsh.model.occ.synchronize()
gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0021)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.0021)
gmsh.model.mesh.generate(gdim)


gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

points=np.array(domain.geometry.x).copy()

triangulation=Triangulation(points[:,0], points[:,1])
triangles=triangulation.triangles


np.random.seed(0)

def idct2(A):
    return idct(idct(A, axis=0, norm='ortho'), axis=1, norm='ortho')


def GRF_sample(points):
    tau=3
    N=256
    xi=np.random.normal(0,1,size=(N,N))
    x = np.linspace(0, 1, num=N)
    y = np.linspace(0, 1, num=N)
    xv, yv = np.meshgrid(x, y)
    alpha=2
    coef=(np.pi**2*(xv**2+yv**2)+tau**2)**(-alpha/2)
    L=N*coef*xi
    L[0,0]=0
    U=idct2(L)
    base_points=(x,y)
    Us=np.reshape(U,(-1,N,N))
    res = map(lambda y: interpolate.interpn(base_points, y, points, method="splinef2d"), Us)
    v=np.vstack(list(res)).astype(np.float64)   
    v_grid=np.abs(Us)
    v=np.abs(v)
    return v,v_grid,base_points


def calculate_simulation(file,t):
    V = fem.FunctionSpace(domain, ("Lagrange", 1))
    f = 1.0
    k = fem.Function(V)
    val,val_grid,grid_points=GRF_sample(points[:,:2])
    k.vector[:]=val
    def on_boundary(x):
        return np.isclose(np.sqrt((x[0]-0.5)**2 + (x[1]-0.5)**2), 0.45)
    boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
    bc = fem.dirichletbc(default_scalar_type(0.), boundary_dofs, V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V) 
    a = k*ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()    
    file.write_function(uh, t)
    u_val=uh.x.array
    t=t+1
    return u_val,val,val_grid,grid_points

np.random.seed(0)
NUM_SAMPLES=600
xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(domain)

uh,val,val_grid,grid_points=calculate_simulation(xdmf,0)
u_data=np.zeros((NUM_SAMPLES,len(uh)))
val_vec=np.zeros((NUM_SAMPLES,len(uh)))
val_grid_vec=np.zeros((NUM_SAMPLES,256,256))
val_grid_vec[0]=val_grid
u_data[0]=uh
val_vec[0]=val

for i in trange(1,NUM_SAMPLES):
    uh,val,val_grid,_=calculate_simulation(xdmf,i)
    u_data[i]=uh
    val_vec[i]=val
    val_grid_vec[i]=val_grid
points=points[:,:2] 
xdmf.close()
np.save("u.npy",u_data)
np.save("v.npy",val_vec)
np.save("triangles.npy",triangles)
np.save("points.npy",points)
np.save("val_grid.npy",val_grid_vec)
np.save("grid_points.npy",grid_points)

points=points[:,:2] 
neigh=[set([]) for i in range(len(points))] 
for i in trange(len(triangles)):
    for j in range(3):
        neigh[triangles[i,j]].add(triangles[i,(j+1)%3])
        neigh[triangles[i,j]].add(triangles[i,(j+2)%3])
    
neigh=[tuple(i) for i in neigh]

int_weights=np.zeros(len(points))
for i in trange(len(triangles)):
    triangle=triangles[i]
    points_tmp=points[triangle]
    points_tmp=np.hstack([points_tmp,np.ones((3,1))])
    for j in range(3):
        int_weights[triangle[j]]+=np.abs(np.linalg.det(points_tmp))/2



num_indices=np.sum([len(neigh[i]) for i in range(len(neigh))])
num_points=len(points)
edges=np.zeros((num_indices+num_points,2))
data_x=np.zeros(num_indices+num_points)
data_y=np.zeros(num_indices+num_points)
h=0
for i in trange(len(points)):
    point_neight=points[list(neigh[i])]
    diff=point_neight-points[i]
    M=np.linalg.inv(diff.T@diff)@diff.T
    Mx=M[0]
    My=M[1]
    tmp_x=0
    tmp_y=0
    for k in range(len(neigh[i])):
        edges[h]=[i,neigh[i][k]]
        data_x[h]=Mx[k]
        data_y[h]=My[k]
        h+=1
        tmp_x-=Mx[k]
        tmp_y-=My[k]

    edges[h]=[i,i]
    data_x[h]=tmp_x
    data_y[h]=tmp_y
    h+=1



diff_matrix_x=csr_matrix((data_x, (edges[:,0], edges[:,1])), shape=(num_points, num_points))
diff_matrix_y=csr_matrix((data_y, (edges[:,0], edges[:,1])), shape=(num_points, num_points))
A_x=diff_matrix_x.dot(diff_matrix_x.T.multiply(int_weights.reshape(-1,1)))
A_y=diff_matrix_y.dot(diff_matrix_y.T.multiply(int_weights.reshape(-1,1)))
sc_matrix=A_x+A_y+csr_matrix(scipy.sparse.diags(int_weights))
save_npz("sc_matrix.npz",sc_matrix)
u_data=np.load("u.npy")
covariance=u_data@(sc_matrix.dot(u_data.T))/NUM_SAMPLES
U,S,V=np.linalg.svd(covariance)
xi=1/np.sqrt(NUM_SAMPLES)*((u_data.T@U))
for i in range(NUM_SAMPLES):
    xi[:,i]=xi[:,i]/np.sqrt(S[i])
coeff=u_data@(sc_matrix.dot(xi))
u_rec=coeff@xi.T

np.save("pod_basis.npy",xi.T)
np.save("pod_coeff.npy",coeff)

import matplotlib.pyplot as plt
import matplotlib.tri as tri

print(np.mean(np.var(u_data,axis=0)))
triang = tri.Triangulation(points[:,0], points[:,1], triangles=triangles)
fig, ax = plt.subplots()
tpc = ax.tripcolor(triang, u_data[-1], shading='flat')
fig.savefig('deeponet.png')