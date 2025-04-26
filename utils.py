import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
import random
import pickle
import os
import time
import csv
from itertools import product

def load_dataset(name):
    if name == 'CSPhD':
        # WARNING!!!! This D_csphd is different from D_csphd.pkl file. They are identical up to permutation tho
        print("Loading CSPhD distance matrix")
        with open('dataset/ca-CSphd.csv', mode ='r') as file:
            csvFile = csv.reader(file)
            celegan_edges = []
            n_lines = 0
            # displaying the contents of the CSV file
            for lines in csvFile:
        #         print(lines)
                if(n_lines > 0):
                    celegan_edges.append(lines)
                n_lines += 1
        G_csphd = nx.Graph()
        for e in celegan_edges:
            if e[0] not in G_csphd.nodes:
                G_csphd.add_node(e[0])
            if e[1] not in G_csphd.nodes:
                G_csphd.add_node(e[0])
            G_csphd.add_edge(e[0], e[1])
        G_csphd = nx.convert_node_labels_to_integers(G_csphd)
        D_csphd = nx.floyd_warshall_numpy(G_csphd)
        # D_celegans = pickle.load(open('dataset/D_celegan.pkl', 'rb'))
        return D_csphd, G_csphd
        # D_csphd = pickle.load(open('dataset/D_csphd.pkl', 'rb'))
        # return D_csphd
    elif name == 'Celegans':
        # WARNING!!!! This D_celegans is different from D_celegan.pkl file. They are identical up to permutation tho
        print("Loading Celegans distance matrix")
        with open('dataset/bio-celegans.csv', mode ='r') as file:
            csvFile = csv.reader(file)
            celegan_edges = []
            n_lines = 0
            # displaying the contents of the CSV file
            for lines in csvFile:
        #         print(lines)
                if(n_lines > 0):
                    celegan_edges.append(lines)
                n_lines += 1
        G_celegan = nx.Graph()
        for e in celegan_edges:
            if e[0] not in G_celegan.nodes:
                G_celegan.add_node(e[0])
            if e[1] not in G_celegan.nodes:
                G_celegan.add_node(e[0])
            G_celegan.add_edge(e[0], e[1])
        G_celegan = nx.convert_node_labels_to_integers(G_celegan)
        D_celegan = nx.floyd_warshall_numpy(G_celegan)
        # D_celegans = pickle.load(open('dataset/D_celegan.pkl', 'rb'))
        return D_celegan, G_celegan
    elif name == 'Karate':
        print("Loading Karate distance matrix")
        G = nx.karate_club_graph()
        D_karate = nx.floyd_warshall_numpy(G)
        return D_karate, G
    elif name == 'Airport':
        print("Loading Airport dataset")
        # Compute distance matrix for Airport
        airport = pickle.load(open('dataset/airport/airport.p', 'rb'))
        # For Airport, we need to do this
        li = [airport.subgraph(c) for c in nx.connected_components(airport)]
        connected_G = nx.Graph(li[0])
        connected_G.remove_edges_from(nx.selfloop_edges(connected_G))
        connected_G = nx.convert_node_labels_to_integers(connected_G)
        D_airport = nx.floyd_warshall_numpy(connected_G)
        return D_airport, connected_G
    elif name == 'Cora':
        print("Loading Cora datset")
        cora = pickle.load(open('dataset/cora/ind.cora.graph', 'rb'))
        # For CORA, we need to do this
        G_cora = nx.from_dict_of_lists(cora)
        li = [G_cora.subgraph(c) for c in nx.connected_components(G_cora)]
        connected_G = nx.Graph(li[0])
        connected_G.remove_edges_from(nx.selfloop_edges(connected_G))
        D_cora = nx.floyd_warshall_numpy(connected_G)
        return D_cora, connected_G
    elif name == 'Disease':
        print("Loading Disease dataset")
        # For disease, use follows. (We have removed header line)
        # opening the CSV file
        with open('dataset/disease_lp/disease_lp.edges.csv', mode ='r') as file:
            csvFile = csv.reader(file)
            disease_lp_edges = []
            n_lines = 0
            # displaying the contents of the CSV file
            for lines in csvFile:
        #         print(lines)
                disease_lp_edges.append(lines)
                n_lines += 1
        disease_lp_G = nx.Graph()
        for e in disease_lp_edges:
            if e[0] not in disease_lp_G.nodes:
                disease_lp_G.add_node(e[0])
            if e[1] not in disease_lp_G.nodes:
                disease_lp_G.add_node(e[0])
            disease_lp_G.add_edge(e[0], e[1])
        disease_lp_G = nx.convert_node_labels_to_integers(disease_lp_G)
        D_disease_lp = nx.floyd_warshall_numpy(disease_lp_G)
        return D_disease_lp, disease_lp_G
    else:
        print("Error")
        return -1, -1
    
# def load_synthetic_dataset(tree, l_steps = 10, n_seed = -1, n_edges = 500, u_delta = 0.1, unweighted = False):
#     # def job(G, D_original, l_steps = 10, n_seed = -1, n_edges = 500, u_delta = 0.1
#     G = tree.copy()
#     D_original = nx.floyd_warshall_numpy(G)
#     N = D_original.shape[0]
# #     If n_seed = -1, then we won't fix the seed. If not, 
#     if(n_seed >= 0):
#         np.random.seed(n_seed)
#     edge_list = []
#     while(len(edge_list) < n_edges):
#         pi = np.random.permutation(N)
#         i, j = pi[0], pi[1]
#         if ((i,j) not in edge_list) and ((j,i) not in edge_list) and (D_original[i,j] > 2.0):
#             edge_list.append((i,j))
#     for idx in range(n_edges):
#         i = edge_list[idx][0]
#         j = edge_list[idx][1]
#         if(unweighted):
#             G.add_edge(i, j, weight = 1)
#         else:
#             G.add_edge(i, j, weight = D_original[i,j] - 2 * u_delta)
#     return G

def count_shortest_paths_bfs(G, source):
    """ Computes shortest path distances and counts using BFS. """
    dist = {node: float('inf') for node in G.nodes}
    count = {node: 0 for node in G.nodes}
    
    dist[source] = 0
    count[source] = 1
    queue = [source]

    for u in queue:
        for v in G.neighbors(u):
            if dist[v] == float('inf'):  # First time visiting
                dist[v] = dist[u] + 1
                count[v] = count[u]
                queue.append(v)
            elif dist[v] == dist[u] + 1:  # Another shortest path
                count[v] += count[u]

    return dist, count

def sample_shortest_path(G, source, target):
    """ Uniformly samples a shortest path from source to target in an unweighted graph. """
    dist, count = count_shortest_paths_bfs(G, source)

    if dist[target] == float('inf'):
        return None, dist[target]  # No path exists

    path = [target]
    current = target

    while current != source:
        predecessors = [v for v in G.neighbors(current) if dist[v] == dist[current] - 1]
        weights = [count[v] for v in predecessors]
        current = random.choices(predecessors, weights=weights)[0]
        path.append(current)

    return path[::-1], dist[target]  # Reverse for correct order

# Generate an Erdős–Rényi random graph
n = 100  # Number of nodes
p = 0.05  # Probability of edge creation
G = nx.erdos_renyi_graph(n, p)

# Choose two random connected nodes
while True:
    source, target = random.sample(list(G.nodes), 2)
    if nx.has_path(G, source, target):
        break

# Sample a shortest path
shortest_path_sample = sample_shortest_path(G, source, target)
print(f"Random shortest path from {source} to {target}: {shortest_path_sample[0]}, with distance {shortest_path_sample[1]}")
# print(len(shortest_path_sample))




def sample_geodesic_triangle(G,p,q,r):
    pq, c = sample_shortest_path(G,p,q)
    qr, a = sample_shortest_path(G,q,r)
    rp, b = sample_shortest_path(G,r,p)
    return pq, qr, rp, c, a, b

def compute_triangle_thinness(D,pq,qr,rp,c,a,b):
    zeta = 0.0
    # a = len(qr) - 1
    # b = len(rp) - 1
    # c = len(pq) - 1
    for i in range(a):
        if 2*i <= a+c-b:
            zeta = max(zeta, D[qr[i], pq[c-i]])
        if 2*i >= a+c-b:
            zeta = max(zeta, D[qr[i], rp[a-i]])
    for j in range(c):
        if 2*j <= b+c-a:
            zeta = max(zeta, D[pq[j], rp[b-j]])
    return zeta

def compute_triangle_slimness(D,pq,qr,rp,c,a,b):
    delta = 0.0
    for v in pq:
        proj_dist = c
        for w in qr+rp:
            proj_dist = min(proj_dist, D[v,w])
        delta = max(delta, proj_dist)
    for v in qr:
        proj_dist = a
        for w in pq+rp:
            proj_dist = min(proj_dist, D[v,w])
        delta = max(delta, proj_dist)
    for v in rp:
        proj_dist = b
        for w in qr+pq:
            proj_dist = min(proj_dist, D[v,w])
        delta = max(delta, proj_dist)
    return delta

def compute_triangle_minsize(D,pq,qr,rp,c,a,b):
    eta = float('inf')
    best_triplet = None
    for x,y,z in product(pq,qr,rp):
        max_dist = max(D[x,y], D[y,z], D[z,x])
        if max_dist < eta:
            eta = max_dist
            best_triplet = (x,y,z)
    return eta

def compute_triangle_insize(D,pq,qr,rp,c,a,b):
    # a = len(qr) - 1
    # b = len(rp) - 1
    # c = len(pq) - 1
    if (a+c-b)%2 == 1:
        iota = float('inf')
        i = int((b+c-a-1)/2)
        j = int((a+c-b-1)/2)
        k = int((a+b-c-1)/2)
        X = [pq[i], pq[i+1]]
        Y = [qr[j], qr[j+1]]
        Z = [rp[k], rp[k+1]]
        for x,y,z in product(X,Y,Z):
            max_dist = max(D[x,y], D[y,z], D[z,x])
            if max_dist < iota:
                iota = max_dist
        return iota + 1
    else:
        x = pq[int((b+c-a)/2)]
        y = qr[int((a+c-b)/2)]
        z = rp[int((a+b-c)/2)]
        return max(D[x,y], D[y,z], D[z,x])

def gp(D,x,y,r):
    return 0.5*(D[x,r] + D[y,r] - D[x,y])

def compute_fourpoint_condition(D,x,y,z,r):
    A = gp(D,x,y,r)
    B = gp(D,y,z,r)
    C = gp(D,x,z,r)
    max_gp = max(A,B,C)
    min_gp = min(A,B,C)
    return (A+B+C) - max_gp - 2 * min_gp

def compute_triangle_stats(G,p,q,r,D):
    pq , qr, rp, c, a, b = sample_geodesic_triangle(G,p,q,r)
    # print(pq, type(pq))
    # pq = list(pq)
    # qr = list(qr)
    # rp = list(rp)
    zeta = compute_triangle_slimness(D,pq,qr,rp,c,a,b)
    tau = compute_triangle_thinness(D,pq,qr,rp,c,a,b)
    eta = compute_triangle_minsize(D,pq,qr,rp,c,a,b)
    iota = compute_triangle_insize(D,pq,qr,rp,c,a,b)
    normalizer = 0.5 * max(1, a, b, c)
    return zeta, tau, eta, iota
    
def gp(d,x,y,r):
    return 0.5*(d[x,r] + d[y,r] - d[x,y])

def fp(d,x,y,z,r):
    A = gp(d,x,y,r)
    B = gp(d,y,z,r)
    C = gp(d,x,z,r)
    max_gp = max(A,B,C)
    min_gp = min(A,B,C)
    return (A+B+C) - max_gp - 2 * min_gp

def get_giant_component(G):
    """Extracts the largest connected component (giant component) from an ER graph."""
    # Get all connected components
    components = list(nx.connected_components(G))
    
    # Find the largest component (giant component)
    giant_component_nodes = max(components, key=len)
    
    # Create subgraph of the giant component
    giant_component = G.subgraph(giant_component_nodes).copy()
    giant_component = nx.convert_node_labels_to_integers(giant_component)
    
    return giant_component

def compute_randomized_hyp_vector(d, i, n_samples = 10000, n_seed = -1):
    hyp_avg = 0.0
    hyp_dist = {}
    N = d.shape[0]
    if(n_seed >= 0):
        np.random.seed(n_seed)
    for cnt in range(n_samples):
        idx = np.random.choice(N-1, 3, replace=False)
        idx = idx + (idx >= i)
        hyp = fp(d, idx[0], idx[1], idx[2], i)
        hyp_avg += hyp
        if hyp not in hyp_dist:
            hyp_dist[hyp] = 1
        else:
            hyp_dist[hyp] += 1
    hyp_avg /= n_samples
    return hyp_avg, hyp_dist

def compute_hyp_vector_for_every_vertices(d, n_samples = 10000, n_seed = -1):
    hyp_avg_list = []
    hyp_dist_list = []
    N = d.shape[0]
    if(n_seed >= 0):
        np.random.seed(n_seed)
    for i in range(N):
        hyp_avg, hyp_dist = compute_randomized_hyp_vector(d, i, n_samples, -1)
        hyp_avg_list.append(hyp_avg)
        hyp_dist_list.append(hyp_dist)
    return hyp_avg_list, hyp_dist_list

# def linkage_to_distance_matrix(Z):
#     N = Z.shape[0] + 1
#     C = []
#     for i in range(N):
#         C.append([i])
#     D = np.zeros((N,N))
#     for i in range(N-1):
#         j = Z[i,0].astype(int)
#         k = Z[i,1].astype(int)
#         for x in C[j]:
#             for y in C[k]:
#                 D[x,y] = Z[i,2]
#                 D[y,x] = Z[i,2]
#         C.append(C[j] + C[k])
#     return D