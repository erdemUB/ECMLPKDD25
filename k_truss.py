import networkx as nx
import sys
import pandas as pd
import time
import json

def coarsen_graph(graph):
    coarse_graph = None
    # pick the node with the highest k-number.
    # find the 2-hop neighbour with highest kcore compute the ktruss weight by following the same steps as shown in Algorithm 1 in "Compression of Weighted Graphs"
    # find the new nodes embedding by averaging using kcore
    return coarse_graph
def load_k_truss(graph_name, graph_edges):
    # load the respective file
    dir = './mtx_files/hierarchy/Hierarchy_raw/23/K_values/'
    filename = dir + graph_name + '.mtx_23_K_values'
    # Read the file into a DataFrame
    df = pd.read_csv(filename, header=None, names=['u', 'v', 'truss'])
    # Convert the columns to the appropriate data types
    df['u'] = df['u'].astype(int)
    df['v'] = df['v'].astype(int)
    df['truss'] = df['truss'].astype(float)
    # Create the dictionary
    degrees = {(row['u'], row['v']): row['truss'] for idx, row in df.iterrows()}
    # load the respective value. set missing values to 0
    degrees_output = []
    for e in graph_edges:
        if (e[0], e[1]) in degrees:
            degrees_output.append(degrees[(e[0], e[1])])
        elif (e[1],e[0]) in degrees:
            degrees_output.append(degrees[(e[1], e[0])])
        else:
            degrees_output.append(0)
    return degrees_output

def get_k_truss(ugraph, graph):
    # find the list of edges in the graph
    edges = [e for e in ugraph.edges]
    # for each edge find the number of triangles it belongs to
    all_cliques = nx.enumerate_all_cliques(ugraph)
    triangles = [x for x in all_cliques if len(x) == 3]

    edge_to_triangle = {}
    for tr in triangles:
        edges = [(tr[i], tr[j]) for i in range(len(tr)) for j in range(i + 1, len(tr))]
        for edge in edges:
            if edge not in edge_to_triangle:
                edge_to_triangle[edge] = []
            edge_to_triangle[edge].append(tr)

    # print(f"Edges to triangles: {edge_to_triangle}")
    # sys.exit()
    degrees = {}
    unproc_edges = set()
    for edge in edge_to_triangle:
        unproc_edges.add(edge)
        degrees[edge] = len(edge_to_triangle[edge])

    degrees = dict(sorted(degrees.items(), key=lambda item: item[1]))
    print("Sorting Complete!")
    for r in degrees:
        if r in unproc_edges:
            # get the list of 3-cliques
            triangles = edge_to_triangle[r]  # triangles related "edge"
            unproc_edges.remove(r)
            # find other unprocessed 2-cliques in the 3-cliques
            edges = []
            for tr in triangles:
                edges += [(tr[i], tr[j]) for i in range(len(tr)) for j in range(i + 1, len(tr)) if
                          (tr[i], tr[j]) in unproc_edges]

            for r_ in edges:
                unproc_edges.remove(r_)
                # reduce their degree conditionally
                if degrees[r_] > degrees[r] and degrees[r_] > 0:
                    degrees[r_] -= 1

    graph_edges = [e for e in graph.edges]
    degrees_output = []

    for e in graph_edges:
        if (e[0], e[1]) in degrees:
            degrees_output.append(degrees[(e[0], e[1])])
        elif (e[1],e[0]) in degrees:
            degrees_output.append(degrees[(e[1], e[0])])
        else:
            degrees_output.append(0)


    return degrees_output