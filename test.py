import matplotlib
matplotlib.use("Agg")

import networkx as nx 
import random
import matplotlib.pyplot as plt

from spectral_embedder import spectral_embedder

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
    plt.savefig(fname + "_original.png")
    # clear previous graph
    plt.clf()

fname = "data/directed_weighted_gnr_graph" # path and filename

# Generate directed, weighted GNR graph
G = nx.gnr_graph(n=20, p=0.35, seed=1234)
for (u,v) in G.edges():
    G[u][v]["weight"] = round(random.uniform(0,1),3)

# Plots G using base spring layout
labels = {}    
for node in G.nodes():
    labels[node] = node
pos = nx.spring_layout(G)
draw_graph(G, pos, fname, labels, weighted=True)

# Embeds G, plots and saves results
spectral_embedder_v4(G, fname, directed=nx.is_directed(G), weighted=True, plot=True, symmetric=False)