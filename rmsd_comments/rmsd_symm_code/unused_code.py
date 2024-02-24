
def draw_graph(xyz, elements, alpha, beta, gamma):
# функция отрисовки молекулы (alpha, beta, gamma - углы поворота)    
    
    clr={ # цветовая схема CPK
    'H': 'lightgray', 'C': 'dimgray', 'N': 'blue', 'O': 'red', 'F': 'green', 
    'Cl': 'green', 'Br': 'darkred', 'I': 'darkviolet', 'He': 'cyan', 
    'Ne': 'cyan', 'Ar': 'cyan', 'Xe': 'cyan', 'Kr': 'cyan', 'P': 'orange', 
    'S': 'yellow', 'B': 'beige', 'Li': 'violet', 'Na': 'violet', 
    'K': 'violet', 'Rb': 'violet', 'Cs': 'violet', 'Be': 'darkgreen', 
    'Mg': 'darkgreen', 'Ca': 'darkgreen', 'Sr': 'darkgreen', 
    'Ba': 'darkgreen', 'Ra': 'darkgreen', 'Ti': 'gray', 'Fe': 'darkorange'}     
    graph = nx.Graph()
    for i, symbol in enumerate(elements):  # добавление узлов графа
      graph.add_node(i) 
      graph.nodes[i]['symbol']=symbol
  
    for i in range(len(elements)):         # добавление рёбер графа
      for j in range(i):
        max_dist = 1.3 * (r[elements[i]] + r[elements[j]])
        if np.linalg.norm(set_xyz[0][i] - set_xyz[0][j]) < max_dist: 
            graph.add_edge(i, j)
        
    rads=[] 
    clrs=[]
    for i in graph.nodes:
      sb=graph.nodes[i]['symbol']
      clrs.append(clr.get(sb, 'pink'))
      rads.append(r.get(sb, 0.5)*3000)    
    # опт. поворот
    xc=xyz.copy() 
    xc-=xyz.mean(axis=0)
    V, _, _ = np.linalg.svd(np.dot(np.transpose(xc),xc)) 
    # доп. поворот
    ra=np.array([[1, 0, 0],[0, np.cos(alpha), -np.sin(alpha)],
                 [0, np.sin(alpha), np.cos(alpha)]])
    rb=np.array([[np.cos(beta), 0, np.sin(beta)],[0, 1, 0],
                  [-np.sin(beta), 0, np.cos(beta)]])
    rg=np.array([[np.cos(gamma), -np.sin(gamma), 0],
                  [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    V=np.dot(V, ra) 
    V=np.dot(V, rb) 
    V=np.dot(V, rg)    
    xy = np.dot(xc, V)[:,:2]
    # позиции номеров атомов
    x_label=xy.copy()
    x_label[:,0]+=0.17
    x_label[:,1]-=0.07 
    
    options = {'node_color': clrs, 'node_size': rads, 'width': 1,}
    plt.figure()
    nx.draw(graph, xy, with_labels = False, style="solid", **options) 
    # окантовка
    ax = plt.gca() 
    ax.collections[0].set_edgecolor("#000000")
    
    node_labels = nx.get_node_attributes(graph, 'symbol')
    # позиции символов атомов
    xy[:,0]-=0.04 
    xy[:,1]+=0.01 
    
    nx.draw_networkx_labels(graph, xy, labels=node_labels, font_weight='bold')
    nx.draw_networkx_labels(graph, x_label, font_size=8)
        
    plt.show() # на экран
#    plt.rc('pgf', texsystem='xelatex') # в LaTeX
#    plt.savefig('/home/alexis/Документы/Диплом_изоморфизм/figure.pgf')

    
def rmsd_base(xyz1, xyz2, elements): 
    # базовый алгоритм xyz  (elements задаёт соотв. элементы)     
    g = nx.Graph() 
    h = nx.Graph()
    for i, symbol in enumerate(elements):  # добавление узлов графа
      g.add_node(i)
      g.nodes[i]['label']=symbol
      h.add_node(i)
      h.nodes[i]['label']=symbol
    for i in range(len(elements)):         # добавление рёбер графа
      for j in range(i):
        max_dist = 1.3 * (ATOMIC_RADII[elements[i]] + ATOMIC_RADII[elements[j]])
        if np.linalg.norm(xyz1[i] - xyz1[j]) < max_dist: g.add_edge(i, j)
        if np.linalg.norm(xyz2[i] - xyz2[j]) < max_dist: h.add_edge(i, j)        
    # поиск изоморфизмов   
    ism = []
    gm = nx.algorithms.isomorphism.GraphMatcher(g, h, 
            node_match=lambda arg1, arg2: arg1['label']==arg2['label'])      
    for i in gm.isomorphisms_iter():
      ism.append([i[j] for j in range(len(i))])   
    # центрирование координат
    xc1=xyz1-xyz1.mean(axis=0) 
    xc2=xyz2-xyz2.mean(axis=0)
    mse_min=float("inf") 
    for j in ism:
        # kabsh algorithm
        C=np.dot(xc1.T, xc2[j,:]) 
        V, _, W = np.linalg.svd(C)
        R = np.dot(V, W)
        x_rot = np.dot(xc1, R)
        d=(xc2[j,:]-x_rot)**2 
        mse=d.sum()/len(d)
        if mse<mse_min: 
            mse_min=mse 
            k=j
            
    return np.sqrt(mse_min), k