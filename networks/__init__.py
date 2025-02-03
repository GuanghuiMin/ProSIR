import networkx as nx
import matplotlib.pyplot as plt
import gzip
import numpy as np
import os


def generate_powerlaw_graph(n, avg_degree, exponent=2.5, seed=None, max_degree=None):
    if seed is not None:
        np.random.seed(seed)

    raw_degrees = np.random.pareto(exponent - 1, n) + 1
    raw_degrees = raw_degrees / np.mean(raw_degrees)
    degrees = np.floor(raw_degrees * avg_degree).astype(int)
    degrees = np.array([max(1, d) for d in degrees])

    if max_degree is not None:
        degrees = np.minimum(degrees, max_degree)

    current_avg = np.mean(degrees)
    scale = avg_degree / current_avg
    degrees = np.floor(degrees * scale).astype(int)
    degrees = np.array([max(1, d) for d in degrees])

    if np.sum(degrees) % 2 != 0:
        degrees[0] += 1

    G = nx.configuration_model(degrees, seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G.to_directed()

def generate_graph(graph_type, nodes=1000, avg_degree=10, rewiring_prob=0.1, seed=10, file_path=None):
    if graph_type == "ba":
        m = avg_degree // 2
        graph = nx.barabasi_albert_graph(nodes, m, seed=seed).to_directed()
    elif graph_type == "er":
        p = avg_degree / (nodes - 1)
        graph = nx.erdos_renyi_graph(nodes, p=p, directed=True, seed=seed)
    elif graph_type == "powerlaw":
        graph = generate_powerlaw_graph(nodes, avg_degree, seed=seed)
    elif graph_type == "smallworld":
        k = avg_degree // 2
        graph = nx.watts_strogatz_graph(nodes, k, rewiring_prob, seed=seed)
        graph = nx.DiGraph(graph)
    else:
        raise ValueError("Invalid graph type or missing parameters.")
    return graph


def load_real_data(data_dir, file_name):
    G = nx.DiGraph()
    file_path = os.path.join(data_dir, file_name)
    print(f"Loading dataset from {data_dir}/{file_name}...")

    with gzip.open(file_path, 'rt') as f:
        if file_name == "com-friendster.top5000.cmty.txt.gz":
            for line in f:
                nodes = list(map(int, line.strip().split()))
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        G.add_edge(nodes[i], nodes[j])
                        G.add_edge(nodes[j], nodes[i])
        elif file_name == "gplus_combined.txt.gz":
            for line in f:
                if line.startswith('#'):
                    continue
                source, target = map(int, line.strip().split())
                G.add_edge(source, target)
        elif file_name in ["soc-sign-bitcoinalpha.csv.gz", "soc-sign-bitcoinotc.csv.gz"]:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(',')
                source, target = int(parts[0]), int(parts[1])
                G.add_edge(source, target)
        else:
            for line in f:
                if line.startswith('#'):
                    continue
                source, target = map(int, line.strip().split())
                G.add_edge(source, target)

    print("Dataset loaded.")
    G = nx.convert_node_labels_to_integers(G, label_attribute="original_id")
    return G


def visualize_graph(graph, title="Directed Graph", nodes_to_draw=500):
    subgraph = graph.subgraph(list(graph.nodes)[:nodes_to_draw])
    plt.figure(figsize=(8, 8))
    nx.draw(subgraph, node_size=10, edge_color='gray', with_labels=False)
    plt.title(f"{title} ({nodes_to_draw} Nodes Subgraph)")
    plt.show()



if __name__ == '__main__':
    graph3 = generate_graph("smallworld", nodes=2000,avg_degree=4, seed=10)
    visualize_graph(graph3, title="Power-law Directed Graph")

