import random
import collections
import random
import time
import signal
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from types import FrameType
from typing import List, Tuple


# Custom exception for handling timeouts
class TimeoutException(Exception):
    """Exception raised when a timeout signal is triggered."""
    pass


def timeout_handler(signum: int, frame: FrameType):
    """
    Signal handler for timeout events. Raises a TimeoutException when a timeout signal is received.

    :param signum: An integer representing the signal number.
    :param frame: A FrameType object representing the current stack frame.
    :raises TimeoutException: An exception raised to interrupt and handle a timeout event.
    """
    raise TimeoutException("Operation timed out")


def configuration_A(S: List[int], seed: int = 42) -> nx.Graph:
    """
    Generates a random graph using the Configuration model with a specified degree sequence.

    :param S: A list of integers where each integer represents the degree of a node.
    :param seed: Seed for the random number generator to ensure reproducibility.
    :return: A networkx Graph object with nodes connected according to the degree sequence S.
    """
    random.seed(seed) # Seed the random number generator for reproducibility
    stubs = [i for i, degree in enumerate(S) for _ in range(degree)]
    n = nx.Graph()
    n.add_nodes_from(range(len(S)))  # Ensure the graph has len(S) nodes
    
    while len(stubs) > 1:  # Need at least 2 stubs to form an edge
        v, w = random.sample(stubs, 2)
        if v != w:  # Avoid self-loops
            n.add_edge(v, w)
            stubs.remove(v)  # Remove v from stubs
            stubs.remove(w)  # Remove w from stubs
    return n


def configuration_B(n: int, P: List[float], seed: int = 42) -> nx.Graph:
    """
    Generates a random graph using the Configuration model with a target degree distribution.

    :param n: The number of nodes in the graph.
    :param P: A list of probabilities representing the degree distribution.
    :param seed: Seed for the random number generator to ensure reproducibility.
    :return: A networkx Graph object configured according to the specified degree distribution.
    """
    random.seed(seed) # Seed the random number generator for reproducibility
    S = [1]  # Initialize with an invalid degree sequence
    while not nx.is_valid_degree_sequence_erdos_gallai(S):
        S = random.choices(population=range(len(P)), weights=P, k=n)
    return configuration_A(S)


def digest_network(network_csv: str, n: int) -> Tuple[nx.Graph, float]:
    """
    Processes a CSV file to create a NetworkX graph, and computes its clustering coefficient
    after simulating a random graph with the same degree distribution.

    :param network_csv: Path to the CSV file containing the network data.
    :param n: Number of nodes in the network.
    :return: A tuple containing the simulated random graph and its average clustering coefficient.
    """
    # Read in the network from the given CSV file
    network = pd.read_csv(network_csv)

    # Create a NetworkX graph from the network data
    network_graph = nx.Graph()
    for idx, row in network.iterrows():
        # Add edges with attributes
        network_graph.add_edge(row['init_node'], row['term_node'], **row.drop(['init_node', 'term_node']).to_dict())

    # Get the degree distribution of the network
    degrees = [network_graph.degree(node) for node in network_graph.nodes()]
    degree_counts = np.bincount(degrees)

    # Generate a random model from the degree distribution
    random_graph = configuration_B(n, degree_counts)

    # Calculate the clustering coefficient of the random graph
    graph_avg = nx.average_clustering(random_graph)

    return random_graph, graph_avg


def analyze_graph(G: nx.Graph):
    """
    Analyzes and prints various properties of the given graph, including its diameter, clustering coefficient,
    assortativity coefficient, and degree distribution.

    :param G: A NetworkX graph object.
    """
    # Check if the graph is connected; this is required to compute the diameter
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        print(f"Diameter of the graph: {diameter}")
    else:
        print("Graph is not connected; diameter of the giant component will be calculated.")
        # Find the largest connected component (giant component)
        giant_component = max(nx.connected_components(G), key=len)
        G_giant = G.subgraph(giant_component).copy()

        # Compute the diameter of the giant component
        diameter = nx.diameter(G_giant)
        print(f"Diameter of the giant component: {diameter}")

    # Compute the average clustering coefficient of the graph
    avg_clustering = nx.average_clustering(G)
    print(f"Average Clustering Coefficient: {avg_clustering}")

    # Compute the degree assortativity coefficient of the graph
    assortativity = nx.degree_assortativity_coefficient(G)
    print(f"Degree Assortativity Coefficient: {assortativity}")

    # Calculate and plot the degree distribution of the graph
    degrees = [G.degree(n) for n in G.nodes()]
    degree_counts = np.bincount(degrees)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(degree_counts)), degree_counts, width=0.80, color='b')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()