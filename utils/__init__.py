import os
import pandas as pd
import networkx as nx
import random
import warnings
import numpy as np
from tqdm import tqdm
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor
import time
from scipy.sparse import isspmatrix, csr_matrix
from scipy.sparse.linalg import eigs

file_title_mapping=dataset_mapping = {
    "soc-sign-bitcoinalpha.csv.gz": "Bitcoin-Alpha",
    "soc-sign-bitcoinotc.csv.gz": "Bitcoin-OTC",
    "p2p-Gnutella08.txt.gz": "p2p-Gnutella",
    "p2p-Gnutella31.txt.gz": "p2p-Gnutella",
    "soc-Epinions1.txt.gz": "soc-Epinions",
    "soc-Slashdot0902.txt.gz": "soc-Slashdot",
    "email-Enron.txt.gz": "email-Enron",
    "email-EuAll.txt.gz": "email-EuAll",
    "soc-pokec-relationships.txt.gz": "soc-Pokec"
}

def save_args_to_csv(args, output_dir):
    args_dict = vars(args)
    args_df = pd.DataFrame(args_dict.items(), columns=["Argument", "Value"])
    args_df.to_csv(os.path.join(output_dir, "args_settings.csv"), index=False)
    print(f"Arguments saved to: {os.path.join(output_dir, 'args_settings.csv')}")

def calculate_centralities_approx(
        graph,
        alpha=0.85,
        max_iter=100,
        tol=1e-6,
        k_for_approx=100,
        seed=42,
        n_jobs=-1,
        methods=None
):
    if methods is None:
        methods = ["Degree", "Eigenvector", "PageRank", "Betweenness", "Closeness"]
    elif isinstance(methods, str):
        methods = [methods]

    results = {}
    n = graph.number_of_nodes()

    def safe_execution(method_name, func):
        try:
            start_time = time.time()
            result = func()
            end_time = time.time()
            print(f"{method_name} completed in {end_time - start_time:.2f} seconds")
            return method_name, result
        except Exception as e:
            print(f"{method_name} failed: {e}")
            return method_name, None

    adj_matrix = nx.adjacency_matrix(graph)

    centrality_functions = {
        "Degree": lambda: {node: deg / float(n - 1)
                           for node, deg in graph.degree()},

        "Eigenvector": lambda: approximate_eigenvector_centrality(
            adj_matrix, max_iter=max_iter, tol=tol
        ),

        "PageRank": lambda: approximate_pagerank(
            graph, alpha=alpha, max_iter=max_iter, tol=tol
        ),

        "Betweenness": lambda: approximate_betweenness(
            graph, k=min(k_for_approx, n), seed=seed
        ),

        "Closeness": lambda: approximate_closeness_centrality(
            graph, k=min(k_for_approx, n), seed=seed
        )
    }

    selected_functions = {name: func
                          for name, func in centrality_functions.items()
                          if name in methods}

    with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        future_to_method = {
            executor.submit(safe_execution, name, func): name
            for name, func in selected_functions.items()
        }

        for future in future_to_method:
            method_name, result = future.result()
            if result is not None:
                results[method_name] = result

    return results

def approximate_eigenvector_centrality(adj_matrix, max_iter=100, tol=1e-6):
    n = adj_matrix.shape[0]
    x = np.ones(n) / np.sqrt(n)

    for _ in tqdm(range(max_iter), desc="Eigenvector iteration"):
        x_next = adj_matrix.dot(x)
        norm = np.linalg.norm(x_next)
        if norm == 0:
            x = np.zeros(n)
            break
        x_next = x_next / norm
        if np.linalg.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next

    return dict(enumerate(x))

def approximate_pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6):
    n = len(graph)
    x = np.ones(n) / n
    adj_matrix = nx.adjacency_matrix(graph)

    out_degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    out_degrees[out_degrees == 0] = 1
    transition_matrix = adj_matrix.multiply(1 / out_degrees[:, np.newaxis])

    for _ in tqdm(range(max_iter), desc="PageRank iteration"):
        x_next = alpha * transition_matrix.dot(x) + (1 - alpha) * np.ones(n) / n
        if np.linalg.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next

    return dict(enumerate(x))

def approximate_betweenness(graph, k=100, seed=None):
    if seed is not None:
        random.seed(seed)

    betweenness = dict.fromkeys(graph, 0.0)
    nodes = list(graph.nodes())
    sampled_nodes = random.sample(nodes, min(k, len(nodes)))

    for s in tqdm(sampled_nodes, desc="Betweenness sampling"):
        S = []
        P = {}
        sigma = dict.fromkeys(nodes, 0.0)
        sigma[s] = 1.0
        D = {}
        Q = [s]
        D[s] = 0

        while Q:
            v = Q.pop(0)
            S.append(v)
            Dv = D[v]
            sigmav = sigma[v]

            for w in graph.neighbors(v):
                if w not in D:
                    Q.append(w)
                    D[w] = Dv + 1
                if D[w] == Dv + 1:
                    sigma[w] += sigmav
                    if w not in P:
                        P[w] = []
                    P[w].append(v)

        delta = dict.fromkeys(nodes, 0)
        while S:
            w = S.pop()
            if w in P:
                for v in P[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    scale = len(sampled_nodes) * (len(nodes) - 1)
    for v in betweenness:
        betweenness[v] /= scale

    return betweenness


def approximate_closeness_centrality(graph, k=100, seed=None):
    if seed is not None:
        random.seed(seed)

    n = len(graph)
    nodes = list(graph.nodes())
    sample_sources = random.sample(nodes, min(k, n))

    dist_matrix = {}
    for s in tqdm(sample_sources, desc="Closeness BFS"):
        distances = nx.single_source_shortest_path_length(graph, s)
        dist_matrix[s] = distances

    closeness = {}
    for v in nodes:
        total_dist = 0
        count = 0
        for s in sample_sources:
            d = dist_matrix[s].get(v, n)
            total_dist += d
            count += 1
        if total_dist > 0:
            closeness[v] = (count - 1) / total_dist
        else:
            closeness[v] = 0.0

    return closeness

def calculate_centralities(graph):
    degree_centrality = nx.degree_centrality(graph)
    eigenvector_centrality = nx.eigenvector_centrality_numpy(graph)
    pagerank = nx.pagerank(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)

    return {
        "Degree": degree_centrality,
        "Eigenvector": eigenvector_centrality,
        "PageRank": pagerank,
        "Betweenness": betweenness_centrality,
        "Closeness": closeness_centrality,
    }
