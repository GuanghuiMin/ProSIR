#!/usr/bin/env python3
import argparse
import os
import gc
import time
import random
import datetime
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times New Roman']

from scipy import sparse
from scipy.stats import kendalltau
from networks import generate_graph, load_real_data
from simulations.mc import run_monte_carlo_simulations
from simulations.pds import pds_prob
from simulations.relaxation_pds import relaxation_pds_prob
from simulations.local_push_pds import localpush_pds_prob
from eval import calculate_top_k_overlap, calculate_infection_probability
from utils import save_args_to_csv, calculate_centralities_approx, file_title_mapping

def calculate_error_bars_for_metrics(mc_infection_probs, method_probs, k):
    kendall_taus = []
    top_k_overlaps = []
    for mc_prob in mc_infection_probs:
        tau, _ = kendalltau(mc_prob, method_probs)
        kendall_taus.append(tau)
        top_k_overlap = calculate_top_k_overlap(method_probs, mc_prob, k)
        top_k_overlaps.append(top_k_overlap)
    return {
        "Kendall Tau": f"{np.mean(kendall_taus):.4f}",
        "Top-K Overlap": f"{np.mean(top_k_overlaps):.4f}"
    }

def calculate_mc_final_s_statistics(mc_final_s_trials):
    mc_final_s_trials = [np.array(x) for x in mc_final_s_trials]
    means = [np.mean(x) for x in mc_final_s_trials]
    lowers = [np.percentile(x, 2.5) for x in mc_final_s_trials]
    uppers = [np.percentile(x, 97.5) for x in mc_final_s_trials]
    mean_s = np.mean(means)
    lower_bound = np.mean(lowers)
    upper_bound = np.mean(uppers)
    mean_error = np.std(means)
    lower_error = np.std(lowers)
    upper_error = np.std(uppers)
    return mean_s, lower_bound, upper_bound, mean_error, lower_error, upper_error

def calculate_error_metrics(mc_infection_probs, method_probs, k):
    kendall_taus = []
    top_k_overlaps = []
    for mc_prob in mc_infection_probs:
        tau, _ = kendalltau(mc_prob, method_probs)
        kendall_taus.append(tau)
        top_k_overlap = calculate_top_k_overlap(method_probs, mc_prob, k)
        top_k_overlaps.append(top_k_overlap)
    return {
        "Kendall Tau": f"{np.mean(kendall_taus):.4f}",
        "Top-K Overlap": f"{np.mean(top_k_overlaps):.4f}"
    }

def save_sparse_results(results, filename):
    if isinstance(results, list):
        sparse.save_npz(filename, results[-1])
    else:
        sparse.save_npz(filename, results)

def load_sparse_results(filename):
    return sparse.load_npz(filename)

def run_single_mc_trial(graph, beta, gamma, initial_infected_nodes, num_simulations, trial_idx, temp_dir):
    start_time = time.time()
    all_final_s_values = []
    all_trajectories_lengths = []
    infection_probs = np.zeros(len(graph))
    batch_size = min(args.mc_batch_size, num_simulations)
    num_batches = (num_simulations + batch_size - 1) // batch_size

    for batch in range(num_batches):
        current_batch_size = min(batch_size, num_simulations - batch * batch_size)
        print(f"[MC Trial {trial_idx+1}] Batch {batch + 1}/{num_batches} ...")
        batch_results, batch_trajectories, _ = run_monte_carlo_simulations(
            graph, beta, gamma, initial_infected_nodes, current_batch_size
        )
        all_final_s_values.extend([traj[-1] for traj in batch_trajectories])
        all_trajectories_lengths.extend([len(traj) for traj in batch_trajectories])
        infection_probs += calculate_infection_probability(batch_results) * (current_batch_size / num_simulations)
        del batch_results, batch_trajectories
        gc.collect()

    runtime = time.time() - start_time
    np.save(os.path.join(temp_dir, f'mc_final_s_trial_{trial_idx}.npy'), all_final_s_values)
    np.save(os.path.join(temp_dir, f'mc_infection_probs_trial_{trial_idx}.npy'), infection_probs)
    return runtime, np.mean(all_trajectories_lengths)

def sparse_dict_to_matrix(dict_data, num_nodes):
    rows = []
    cols = []
    data = []
    if 'iteration' in dict_data:
        dict_data.pop('iteration')
    state_to_col = {'susceptible': 0, 'infected': 1, 'recovered': 2}
    for state, node_dict in dict_data.items():
        if state in state_to_col:
            for node, value in node_dict.items():
                if isinstance(node, (int, str)) and str(node).isdigit():
                    rows.append(int(node))
                    cols.append(state_to_col[state])
                    data.append(float(value))
    return sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, 3))

def matrix_to_dict(sparse_matrix):
    states = ['susceptible', 'infected', 'recovered']
    result = {state: {} for state in states}
    for i, j in zip(*sparse_matrix.nonzero()):
        value = sparse_matrix[i, j]
        if value != 0:
            result[states[j]][str(i)] = float(value)
    return result

def run_method_trial(method_func, graph, beta, gamma, initial_infected_nodes,
                     tol, trial_idx, temp_dir, *method_args):
    start_time = time.time()
    results = None
    try:
        results = method_func(graph, beta, gamma, initial_infected_nodes, tol, *method_args)
        runtime = time.time() - start_time
        if not results or not isinstance(results, list):
            raise ValueError(f"Method returned invalid results: {results}")

        final_state = results[-1]
        num_nodes = len(graph)
        sparse_final_state = sparse_dict_to_matrix(final_state, num_nodes)
        save_path = os.path.join(temp_dir, f'method_results_trial_{trial_idx}.npz')
        save_sparse_results(sparse_final_state, save_path)
        final_s = sum(float(v) for v in final_state.get('susceptible', {}).values())
        return {
            "runtime": runtime,
            "final_s": final_s,
            "iterations": len(results),
            "final_state": final_state
        }
    except Exception as e:
        print(f"[Error] Method trial {trial_idx} failed: {e}")
        raise
    finally:
        if results is not None:
            del results
        gc.collect()

def load_method_final_probs(trial_idx, temp_dir, num_nodes):
    sparse_final_state = load_sparse_results(
        os.path.join(temp_dir, f'method_results_trial_{trial_idx}.npz')
    )
    final_dict = matrix_to_dict(sparse_final_state)
    method_probs = np.zeros(num_nodes)
    for node_str in final_dict['infected']:
        node = int(node_str)
        method_probs[node] += final_dict['infected'][node_str]
    for node_str in final_dict['recovered']:
        node = int(node_str)
        method_probs[node] += final_dict['recovered'][node_str]
    return method_probs

def get_infection_prob_1_minus_s(final_state, num_nodes):
    infection_prob = np.zeros(num_nodes, dtype=np.float32)
    s_dict = final_state.get('susceptible', {})
    for node_str, val_s in s_dict.items():
        node_id = int(node_str)
        infection_prob[node_id] = 1.0 - float(val_s)
    return infection_prob

def plot_threshold_curves_with_mc_horizontal_band(
        mc_infected_fraction_list,
        pds_probs,
        relax_probs,
        localpush_probs,
        num_nodes,
        output_path="curve_vs_mc_baseline.png",
        num_thresholds=200,
        file_name="",
):
    mc_mean = np.mean(mc_infected_fraction_list)
    mc_lower = np.percentile(mc_infected_fraction_list, 2.5)
    mc_upper = np.percentile(mc_infected_fraction_list, 97.5)
    thresholds = np.linspace(0, 1, num_thresholds)

    def fraction_infected_curve(p, t_array):
        return np.array([np.mean(p >= t) for t in t_array])

    pds_curve = fraction_infected_curve(pds_probs, thresholds) if pds_probs is not None else None
    relax_curve = fraction_infected_curve(relax_probs, thresholds) if relax_probs is not None else None
    lp_curve = fraction_infected_curve(localpush_probs, thresholds) if localpush_probs is not None else None

    def fraction_infected_runs(mc_fractions, t_array):
        return np.array([np.mean(mc_fractions >= t) for t in t_array])

    mc_scan_curve = fraction_infected_runs(mc_infected_fraction_list, thresholds)

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({
        'axes.labelsize': 16,
        'axes.titlesize': 24,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })

    plt.fill_between(
        thresholds,
        [mc_lower] * len(thresholds),
        [mc_upper] * len(thresholds),
        color='gray', alpha=0.3,
        label=f"MC 2.5%-97.5%: [{mc_lower:.3f}, {mc_upper:.3f}]"
    )
    plt.plot(thresholds, [mc_mean] * len(thresholds), 'k-', lw=4, label=f"MC mean = {mc_mean:.3f}")
    if pds_curve is not None:
        plt.plot(thresholds, pds_curve, 'b-', label="PDS", lw=3)
    if relax_curve is not None:
        plt.plot(thresholds, relax_curve, 'r--', label="Relaxation-PDS", lw=3)
    if lp_curve is not None:
        plt.plot(thresholds, lp_curve, 'g-.', label="LocalPush-PDS", lw=3)

    plt.axvline(x=0.5, color='k', linestyle='--', label="t=0.5", lw=5)
    plt.xlabel("Threshold", fontsize=24)
    plt.ylabel("Infected fraction ratio", fontsize=24)
    plt.title(file_title_mapping.get(file_name, ""), fontsize=30)
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    pdf_output_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_output_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[Info] Figure saved to: {pdf_output_path}")

def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    print("[Info] Loading or generating graph...")
    if args.file_name:
        graph = load_real_data(args.data_dir, args.file_name)
    else:
        graph = generate_graph(graph_type=args.graph_type,
                               avg_degree=args.average_degree,
                               nodes=args.nodes)
    if graph.is_multigraph():
        graph = nx.DiGraph(graph)
    args.nodes = len(graph.nodes())
    args.average_degree = len(graph.edges()) / len(graph.nodes())
    initial_infected = max(round(args.initial_infected_ratio * args.nodes), 1)
    save_args_to_csv(args, output_dir)
    if len(graph.nodes()) < initial_infected:
        raise ValueError("The graph does not have enough nodes for the initial infected nodes.")
    initial_infected_nodes = random.sample(list(graph.nodes()), initial_infected)

    print("[Info] Running Monte Carlo simulation (single trial) ...")
    mc_runtime, mc_iterations = run_single_mc_trial(
        graph, args.beta, args.gamma, initial_infected_nodes,
        args.num_simulations, 0, temp_dir
    )
    print(f"[MC] Finished. Runtime={mc_runtime:.2f}s, Iterations={mc_iterations:.2f} (avg).")

    print("[Info] Loading MC trial results from disk...")
    mc_final_s_trials = [
        np.load(os.path.join(temp_dir, f'mc_final_s_trial_0.npy'))
    ]
    mc_infection_probs = [
        np.load(os.path.join(temp_dir, f'mc_infection_probs_trial_0.npy'))
    ]
    print("[Info] Calculating MC summary statistics...")
    mean_s, lower_bound, upper_bound, mean_error, lower_error, upper_error = \
        calculate_mc_final_s_statistics(mc_final_s_trials)
    mc_summary = {
        "Metric": [
            "Mean S",
            "Lower Bound (2.5%)",
            "Upper Bound (97.5%)",
            "MC Runtime (s)",
            "MC Iterations"
        ],
        "Value": [
            f"{mean_s:.4f}",
            f"{lower_bound:.4f}",
            f"{upper_bound:.4f}",
            f"{mc_runtime:.4f}",
            f"{mc_iterations:.4f}"
        ]
    }
    mc_summary_df = pd.DataFrame(mc_summary)
    mc_summary_df.to_csv(os.path.join(output_dir, "mc_summary.csv"), index=False)
    print("[Info] MC summary saved to mc_summary.csv")

    mc_final_s = np.load(os.path.join(temp_dir, "mc_final_s_trial_0.npy"))
    mc_infected_fraction_list = 1 - (np.array(mc_final_s) / args.nodes)

    methods = {
        "PDS": pds_prob,
        "Relaxation-PDS": relaxation_pds_prob,
        "LocalPush-PDS": localpush_pds_prob,
    }
    method_final_states = {}
    num_nodes = args.nodes
    for method_name, method_func in methods.items():
        print(f"[Method] {method_name} ...")
        if method_name == "Relaxation-PDS":
            trial_result = run_method_trial(
                method_func, graph, args.beta, args.gamma,
                initial_infected_nodes, args.tol, 0, temp_dir,
                args.omega
            )
        elif method_name == "LocalPush-PDS":
            trial_result = run_method_trial(
                method_func, graph, args.beta, args.gamma,
                initial_infected_nodes, args.tol, 0, temp_dir,
                args.p
            )
        else:
            trial_result = run_method_trial(
                method_func, graph, args.beta, args.gamma,
                initial_infected_nodes, args.tol, 0, temp_dir
            )
        print(f"    -> Finished. Runtime={trial_result['runtime']:.2f}s, Iterations={trial_result['iterations']}")
        method_final_states[method_name] = trial_result.get("final_state")

        method_probs = load_method_final_probs(trial_idx=0, temp_dir=temp_dir, num_nodes=num_nodes)
        error_metrics = calculate_error_metrics(mc_infection_probs, method_probs, k=round(0.1 * args.nodes))
        trial_data = {
            "Method": [method_name],
            "Runtime (s)": [f"{trial_result['runtime']:.4f}"],
            "Kendall Tau": [error_metrics['Kendall Tau']],
            "Top-K Overlap": [error_metrics['Top-K Overlap']]
        }
        method_df = pd.DataFrame(trial_data)
        method_filename = f"{method_name.replace(' ', '_')}_summary.csv"
        method_df.to_csv(os.path.join(output_dir, method_filename), index=False)
        print(f"[Info] Saved {method_name} results to {method_filename}")

    metrics = {
        "Kendall Tau": {},
        "Top-K Overlap": {}
    }
    mean_mc_infection_prob = np.mean(mc_infection_probs, axis=0)
    print("[Info] Calculating centralities...")
    centralities = calculate_centralities_approx(graph)
    for name, values in centralities.items():
        print(f"[centrality] Processing {name} centrality...")
        centrality_probs = np.array([values[node] for node in range(len(mean_mc_infection_prob))])
        metrics_with_error = calculate_error_bars_for_metrics(mc_infection_probs, centrality_probs, k=round(args.top_k_ratio * args.nodes))
        metrics["Kendall Tau"][name] = metrics_with_error["Kendall Tau"]
        metrics["Top-K Overlap"][name] = metrics_with_error["Top-K Overlap"]
    print("[Info] Centralities calculated.")
    results_df = pd.DataFrame(metrics).T
    results_csv_path = os.path.join(output_dir, "centralities.csv")
    results_df.to_csv(results_csv_path)
    print(f"[Info] Results saved to: {results_csv_path}")


    pds_probs = get_infection_prob_1_minus_s(method_final_states["PDS"], args.nodes)
    relax_probs = get_infection_prob_1_minus_s(method_final_states["Relaxation-PDS"], args.nodes)
    lp_probs = get_infection_prob_1_minus_s(method_final_states["LocalPush-PDS"], args.nodes)

    fig_path = os.path.join(output_dir, "mc_baseline_vs_curves.png")
    plot_threshold_curves_with_mc_horizontal_band(
        mc_infected_fraction_list=mc_infected_fraction_list,
        pds_probs=pds_probs,
        relax_probs=relax_probs,
        localpush_probs=lp_probs,
        num_nodes=args.nodes,
        output_path=fig_path,
        num_thresholds=200,
        file_name=args.file_name
    )

    print("[Info] Cleaning up temporary files...")
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    print(f"[Done] All results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIR model simulations.")
    parser.add_argument("--graph_type", type=str, default="powerlaw", help="Graph type (ba, er, powerlaw, smallworld)")
    parser.add_argument("--nodes", type=int, default=10000, help="Number of nodes in the graph")
    parser.add_argument("--average_degree", type=float, default=5, help="Average degree of the graph")
    parser.add_argument("--beta", type=float, default=1/18, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=1/9, help="Recovery rate")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for convergence")
    parser.add_argument("--num_simulations", type=int, default=50, help="Number of Monte Carlo simulations")
    parser.add_argument("--mc_batch_size", type=int, default=20, help="Batch size of Monte Carlo simulations")
    parser.add_argument("--initial_infected_ratio",default=0.01, type=float, help="Ratio of initially infected nodes in the population")
    parser.add_argument("--top_k_ratio", default=0.1, type=float, help="Top-K ratio for overlap calculation")
    parser.add_argument("--omega", type=float, default=1.3, help="Relaxation rate for Relaxation-PDS")
    parser.add_argument("--p", type=int, default=10, help="Number of pre-heats steps in LocalPush-PDS")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to load real data")
    parser.add_argument("--file_name", default="", type=str, help="File name of the real data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")
    args = parser.parse_args()
    main(args)