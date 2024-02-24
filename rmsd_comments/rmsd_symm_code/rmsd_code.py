# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:09:31 2024
@author: Michael Granzev
"""

import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt

# радиусы атомов
ATOMIC_RADII = {
  'H':  0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B':  0.84, 'C':  0.69, 
  'N':  0.71, 'O':  0.66, 'F':  0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 
  'Al': 1.21, 'Si': 1.11, 'P':  1.07, 'S':  1.05, 'Cl': 1.02, 'Ar': 1.06, 
  'K':  2.03, 'Ca': 1.76, 'Sc': 1.7,  'Ti': 1.6,  'V':  1.53, 'Cr': 1.39, 
  'Mn': 1.61, 'Fe': 1.52, 'Co': 1.5,  'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 
  'Ga': 1.22, 'Ge': 1.2,  'As': 1.19, 'Se': 1.2,  'Br': 1.2,  'Kr': 1.16, 
  'Rb': 2.2,  'Sr': 1.95, 'Y':  1.9,  'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 
  'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 
  'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I':  1.39, 'Xe': 1.4, 
  'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 
  'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 
  'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9,  'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 
  'Ta': 1.7,  'W':  1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 
  'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4, 
  'At': 1.5,  'Rn': 1.5,  'Fr': 2.6,  'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 
  'Pa': 2.0,  'U':  1.96, 'Np': 1.9,  'Pu': 1.87, 'Am': 1.8,  'Cm': 1.69
}


def load_set(N_atoms, N_samples, file_name):    
# функция загрузки тестовых данных
  
  # data = open(file_name,"r").readlines()
  # NK: не сделал f.close()
  with open(file_name,"r") as f: # <- 'with' automatically runs f.close() 
    data = f.readlines()
  set_xyz=[]
  set_el=[]
  for i in range(N_samples):
    xyz=np.zeros((N_atoms,3))
    elements=[]
    for j in range(len(xyz)):
     part=data[i*(N_atoms+2)+j+2].split()
     elements.append(part[0]) 
     xyz[j,:]=np.array(part)[1:]
    set_xyz.append(xyz)  
    set_el.append(elements)
    assert xyz.shape[0] == len(elements)
  return set_xyz, set_el

def hungarian(x, y):
# венгерский алгоритм    
  n=len(x)
  R=np.zeros([n,n])
  for i in range(n):
      for j in range(n):
          R[i,j]=np.sum((x[i,:]-y[j,:])**2)    
  n=len(R)+1 
  m=len(R[0])+1
  u=[0]*n 
  v=[0]*m 
  p=[0]*m
  way=[0]*m
  for i in range (1,n):
    us=[False]*m 
    minv=[np.Inf]*m 
    p[0]=i 
    j0=0
    while True:
      us[j0]=True
      i0=p[j0] 
      d=np.Inf
      for j in range(1,m):
        if us[j]==False:
          c=R[i0-1, j-1]-u[i0]-v[j]
          if c<minv[j]: 
                minv[j]=c
                way[j]=j0
          if minv[j]<d: 
            d=minv[j]
            j1=j
      for j in range(0,m):
        if us[j]==True: 
          u[p[j]]+=d
          v[j]-=d
        else: 
            minv[j]-=d
      j0=j1
      if p[j0]==0: 
            break
    while True:
      j1=way[j0]
      p[j0]=p[j1] 
      j0=j1
      if j0==0: 
            break
    
  return [p[i]-1 for i in range(1,m)]

def calc(xyz1, xyz2, groups, ism):
# функция вычисления rmsd    
  x0=xyz1.copy() 
  x=xyz2.copy()
  # начальный поворот
  mse_min=float(np.inf)
  for i in range(len(ism)):     
    for j in groups[i]: # усреднение 
      x0[j,:]=x0[j,:].mean(axis=0) 
      x[j,:]=x[j,:].mean(axis=0) 
    C=np.dot(x.T, x0[ism[i],:]) 
    V, _, W = np.linalg.svd(C)    
    R = np.dot(V, W) 
    xr = np.dot(x, R) 
    d=(x0[ism[i],:]-xr)**2
    mse=np.sum(d)
    if mse<mse_min:
      mse_min=mse
      R0=R.copy()
      i0=ism[i].copy()
  # неусреднённые данные          
  x0=xyz1.copy() 
  x=xyz2.copy()
  xr = np.dot(x, R0)
  g=groups[i].copy()    
  ind0=np.zeros(N_atoms, dtype=np.int64)
  n=0   
  while True: 
    ind1=np.array(i0.copy(), dtype=np.int64)
    for j in g:
      a=x0[ind1[j],:].copy()
      b=xr[j,:].copy()
      c=hungarian(a, b)
      t=ind1[j]
      ind1[j]=t[c]      
    C=np.dot(x.T, x0[ind1,:]) 
    V, _, W = np.linalg.svd(C)   
    R = np.dot(V, W) 
    xr = np.dot(x, R)  
    d=(x0[ind1,:]-xr)**2      
    if list(ind1) == list(ind0): 
        break
    ind0=ind1.copy()  
    n+=1 
    
  return np.sqrt(d.sum()/len(d)), n 


#   Начало программы
if __name__ == "__main__": # NK: See https://www.youtube.com/watch?v=g_wlZ9IhbTs
  N_atoms=27     # число атомов
  N_samples=701  # число примеров
  rel=0.03       # отн. ошибка начального приближения
  t=time.time()  # запуск таймера

  # загрузка данных
  set_xyz, set_el = load_set(N_atoms, N_samples, 'initial_set.xyz')
  elements=set_el[0]

  # центрирование данных
  x=np.zeros([N_atoms,3]) # NK: N_atoms instead of 27?
  for i in range(len(set_xyz)):
    x=set_xyz[i]-set_xyz[i].mean(axis=0)
    set_xyz[i]=x
    # NK: вполне мог сделать set_xyz[i] -= set_xyz[i].mean(axis=0)

  # построить граф молекулы
  g = nx.Graph()
  for i, symbol in enumerate(elements):  # добавление узлов графа
    g.add_node(i)
    g.nodes[i]['label'] = symbol
  for i in range(len(elements)):         # добавление рёбер графа
    for j in range(i):
      max_dist = 1.3 * (ATOMIC_RADII[elements[i]] + ATOMIC_RADII[elements[j]])
      if np.linalg.norm(set_xyz[0][i] - set_xyz[0][j]) < max_dist:
        g.add_edge(i, j) # NK: Why was this on the same line with 'if'?
    
  # поиск перестановок автоморфизмов  
  gm = nx.algorithms.isomorphism.GraphMatcher(g, g, 
            node_match=lambda arg1, arg2: arg1['label'] == arg2['label'])   
  ism = []   
  for i in gm.isomorphisms_iter():
    ism.append([i[j] for j in range(len(i))])
  ism=[np.array(ism)] # NK: Prefer meaningful variable names

  # группировка перестановок
  n0=3/(1-(1-3/N_atoms)*((1-rel)**2)) # мне не очевидно, что это за формула. "отн. ошибка начального приближения" тоже мало что говорит
  n_min=0 # число атомов, не эквивалентных никому кроме себя

  # NK: Could use something like this:
  atom_symmtetry_orders = [
    len(set(ism[0][:,i]))
    for i in range(N_atoms)
  ]
  # then,
  # n_min = atom_symmtetry_orders.count(1)

  for k in ism:
    for i in range(N_atoms):
      if len(set(k[:,i])) == 1:
        n_min+=1
  while n_min<n0:
    isn=[] # NK: Prefer meaningful variable names
    for k in ism:
      ind=[] # NK: Prefer meaningful variable names
      for i in range(N_atoms):
        if len(set(k[:,i])) == 2: # NK: Подозрительно часто фигурирует симметрийное число 2, а что в нём особенного?
          ind.append(i)

      n_max=0
      for i in range(len(ind)):
        n=0  
        for j in range(len(ind)):
          if len(set(k[:,ind[i]]-k[:,ind[j]]))==2:
            n+=1
        if n>n_max:
          n_max=n
          i_max=ind[i]
      iss = k[k[:, i_max].argsort()] # NK: Prefer meaningful variable names
      isn.append(iss[0:len(iss)//2,:])
      isn.append(iss[len(iss)//2:len(iss),:]) 

    ism=isn.copy() 
    n_min=np.inf
    for k in ism:
      n=0
      for i in range(N_atoms):
        if len(set(k[:,i])) == 1:
          n+=1
      if n<n_min:
        n_min=n

  # выделение групп автоморфизмов
  gr=[]
  ip=[]
  for k in ism:
    ind=[]
    atm=[]
    for i in range(N_atoms):
      if len(set(k[:,i])) > 1:
        ind.append(i) 
    while len(ind)>0:  
      i=[]
      a=set()
      i0=ind[0]
      while True:
        i.append(i0)
        ind.remove(i[-1])  
        for j in range(len(k)):
          a=a | set(np.where (k==k[j,i[-1]])[1])
        a=a.difference(set(i)) 
        if len(a)==0: 
          atm.append(i) 
          break    
        i0=list(a)[0]
    gr.append(atm) 
    ip.append(k[0,:])
  
  # вычисление матрицы rmsd  
  M=np.zeros([N_samples, N_samples]) # NK: Prefer meaningful variable names
  for i in range(N_samples-1):
    print(i)
    for j in range(i+1, N_samples):
        M[i,j], n = calc(set_xyz[i], set_xyz[j], gr, ip) 
        M[j,i]=M[i,j]
  print(time.time()-t) # время работы программы

  # загрузка данных для сравнения
  # NK: из нашей переписки:
  #  """
  #  Вот тут матрица RMSD 701x701. Её можно загрузить в numpy командой
  #  matrix = np.loadtxt('rmsd_matrix.txt') <- одна строчка вместо пяти
  #  """

  # Лучше так не писать:
  # data = open('rmsd_matrix.txt',"r").readlines()
  # => забыл f.close() !!

  with open('rmsd_matrix.txt',"r") as f:
    data = f.readlines()
  s=[]
  for j in range(N_samples): 
    s.append(data[j].split())
  res=np.array(s, dtype=np.double)   

  # вычисление ошибок  
  err=[]
  for i in range(N_samples-1):
    for j in range(i+1, N_samples):
      err.append(M[i,j]-res[i,j])

  # гистограмма распределения ошибок    
  fig = plt.figure() 
  ax = plt.axes()
  ax.hist(err, 30) 
  ax.set_xlabel("value") 
  ax.set_ylabel("N errors")
  ax.grid()
