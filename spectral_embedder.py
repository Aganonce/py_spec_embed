import matplotlib
matplotlib.use("Agg")

import networkx as nx 
import numpy as np 
import scipy.linalg as sp
import matplotlib.pyplot as plt

random_seed = 1234
np.random.seed(random_seed)

def rescale_layout(pos, scale = 1):
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos

def draw_graph(G, pos, fname, labels, weighted = False):
    # draw networkx graph
    nx.draw_networkx_nodes(G, pos, node_size = 200)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels = labels, font_size = 12)
    if (weighted):
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
    # resize graph
    fig = plt.gcf()
    fig.set_size_inches((14, 14), forward = False)
    plt.savefig(fname + "_embedded.png")
    # clear previous graph
    plt.clf()

def save_positions(G, pos, fname, labels, weighted = False, plot = False):
    # save node positions
    f = open(fname + "_node_pos.txt", "w")
    for key, item in pos.items():
        f.write(str(key) + " " + str(item[0]) + " " + str(item[1]) + "\n")
    f.close()

    # if weighted, save edgelist with weights
    if (weighted):
        f = open(fname + "_weighted_edges.txt", "w")
    else:
        f = open(fname + "_edges.txt", "w")
    edge_labels = nx.get_edge_attributes(G, "weight")
    for key, item in edge_labels.items():
        f.write(str(key[0]) + " " + str(key[1]) + "\n")
    f.close()
    
    # plot embedded network
    if (plot):
        draw_graph(G, pos, fname, labels, weighted)
    

def undirected_embedder(G, fname, directed, weighted, plot, dim):
    labels = {}    
    for node in G.nodes():
        labels[node] = node

    try:
        pos = nx.spectral_layout(G)
        save_positions(G, pos, fname, labels, weighted, plot)
    except:
        print("NOTE: Advanced calculation exceeded computation time. Using base calculation...")

        # adj matrix
        A = nx.adj_matrix(G)
        A = A.todense()
        A = np.array(A, dtype = np.float64)

        if (directed):
            A = A + np.transpose(A)

        # get degree matrix
        D = np.diag(np.sum(A, axis=0))

        # calculate combinatorial Laplacian
        L = D - A

        D_inv = np.linalg.inv(D)

        # calculate random walk Laplacian (recommended)
        L_rw = D_inv * L

        # calculate symmetric Laplacian
        # L_sym = D_sr * L * D_sr

        eig_val, eig_vec = np.linalg.eig(L_rw)

        index = np.argsort(eig_val)[1:dim + 1]

        pos = np.real(eig_vec[:, index])

        pos = rescale_layout(pos)

        count = 0
        dict_pos = {}
        for node in G.nodes():
            dict_pos[node] = np.array([pos[count][0], pos[count][1]])
            count += 1

        save_positions(dict_pos, fname, labels, weighted, plot)

def spectral_embedder(G, fname, directed=False, weighted=False, symmetric=False, reverse=False, plot=False, alpha=0.35, dim=2):
    """Embed nodes, edges in Euclidean space using Laplacian

    Parameters
    ----------
    G : Networkx graph. Must include weights. If the graph is unweighted, set weights to 1 prior.

    fname : Filename for output graph or txt files. Do not include file type.

    directed : bool (default: False)
        Indicating if edgelist is directed or undirected

    weighted : bool (default: False)
        Indicating if edgelist is weighted or unweighted. If weighted, weights will be saved 
        in an edgelist output.
        
    symmetric : bool (default: False)
        For directed graphs. Symmetrizes directed graph in order to perform undirected transformation.

    reverse : bool (default: False)
        Reverses direction for directed graphs. Used to flip flow of information importance.

    plot : bool (default: False)
        Plots embedded networks along with txt files.

    alpha : int (default: 0.35)
        Random walk "teleport" probability.

    dim : int (default: 2)
        Dimension of layout. (2 recommended.)

    Returns
    -------
    None : Saves two txt files (at fname) with nodes and their euclidean positions, and an 
    edgelist with their weights (if weighted). Plots and saves embedded network (at fname) if plot is True.
    """

    if (directed):
        if (reverse):
            G = G.reverse(copy = True)

        if (symmetric):
            undirected_embedder(G, fname, directed, weighted, plot, dim)
        else:
            labels = {}    
            for node in G.nodes():
                labels[node] = node

            # get adj matrix
            A = nx.adj_matrix(G)
            A = A.todense()
            A = np.array(A, dtype = np.float64)

            M, N = A.shape

            # get degree matrix
            D = np.diag(np.sum(A, axis=0))

            # get initial random walk matrix
            try:
                R = np.dot(A, np.linalg.inv(D))
            except:
                R = np.zeros((N, N))
                for i in range(len(D)):
                    d = D[i][i]
                    if (d > 0):
                        for j in range(len(D)):
                            R[j][i] = A[j][i] / d

            J = np.ones((N, N))

            # rwm with google trick
            R_bar = (1 - alpha) * R + (alpha / N) * J

            R_bar_prime = R_bar.T

            # calculate pi
            V, pi_prime = np.linalg.eig(R_bar)

            pi_prime = np.matrix(pi_prime)

            pi = np.diagflat(pi_prime.diagonal())

            I = np.identity(N)

            sqrt_pi = sp.sqrtm(pi)

            inv_sqrt_pi = np.linalg.inv(sp.sqrtm(pi))

            # calculate symmetric Laplacian
            L_symm = I - (np.dot(np.dot(sqrt_pi, R_bar), inv_sqrt_pi) + np.dot(np.dot(inv_sqrt_pi, R_bar_prime), sqrt_pi)) / 2

            # get nontrivial eigenvectors
            eig_val, eig_vec = np.linalg.eig(L_symm)

            index = np.argsort(eig_val)[1:dim + 1]

            pos = np.real(eig_vec[:, index])

            pos = rescale_layout(pos)

            count = 0
            dict_pos = {}
            for node in G.nodes():
                dict_pos[node] = np.array([pos[count][0], pos[count][1]])
                count += 1

            save_positions(G, dict_pos, fname, labels, weighted, plot)
    else:
        if (symmetric):
            print("NOTE: Symmetric can only be used on directed graphs.")

        if (reverse):
            print("NOTE: Reverse can only be used on directed graphs.")

        undirected_embedder(G, fname, directed, weighted, plot, dim)