import networkx as nx
import random
import time
import signal
from typing import List, Tuple
from utilities import TimeoutException, timeout_handler

# NOTE:
# The functions below are the implementations of the models described in the project.
# To use these functions in conjunctions with the utilities.py file as they are now,
# you must supply the number of nodes and the clustering coefficient of the network you're trying to examine.

# Model 1
def model1(G: nx.Graph, c_current: float, cluster: float, allowed_error: float, 
           nodes_per_round: int, seed: int = 42) -> Tuple[nx.Graph, int, List[float]]:
    """
    Modifies a graph to achieve a target clustering coefficient within a specified error margin,
    subject to a maximum runtime.

    :param G: A networkx graph on which operations will be performed.
    :param c_current: Current clustering coefficient of the graph.
    :param cluster: Target clustering coefficient.
    :param allowed_error: Acceptable margin of error for clustering coefficient.
    :param nodes_per_round: Number of nodes to consider for new connections each round.
    :param seed: Seed for the random number generator to ensure reproducibility.
    :return: A tuple containing the modified graph, the number of iterations completed, and
             the history of clustering coefficients as a list.
    """
    random.seed(seed)  # Seed the random number generator for reproducibility
    c_steps = [c_current]  # Tracks changes of clustering coefficient
    i = 0  # Iteration counter
    cluster_bound = [cluster - allowed_error, cluster + allowed_error]  # Clustering bounds
    start_time = time.time() # Record the start time

    # Setup timeout handler to avoid infinite execution
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3600)  # Set the alarm for 60 minutes

    try:
        while c_current < cluster_bound[0] or c_current > cluster_bound[1]:
            i += 1

            for node in G.nodes():
                current_links = list(G[node]) # Access current links as a list

                if current_links:
                    link_to_remove = random.choice(current_links) # Randomly select one link to remove
                    G.remove_edge(node, link_to_remove) # Remove the selected link

                # Potential new links, excluding current neighbors and the node itself
                possible_new_links = [n for n in G.nodes() if n not in current_links and n != node]

                # Samples nodes for potential connection if there are more than enough candidates
                if len(possible_new_links) > nodes_per_round:
                    selected_nodes = random.sample(possible_new_links, nodes_per_round)
                else:
                    selected_nodes = possible_new_links # Use all possible links

                c_possible = []

                for possible in selected_nodes:
                    G.add_edge(node, possible)  # Temporarily add the new link
                    c_possible.append(nx.average_clustering(G))  # Calculate new clustering coefficient
                    G.remove_edge(node, possible)  # Remove the temporary link

                # Select the best new link if there are possible new links
                if c_possible:
                    best_node = selected_nodes[c_possible.index(max(c_possible))]
                    G.add_edge(node, best_node)

            c_current = nx.average_clustering(G)  # Update the clustering coefficient
            c_steps.append(c_current)  # Track the clustering coefficient progression

        signal.alarm(0)  # Disable the alarm
        execution_time = time.time() - start_time # Calculate total execution time
        print(f"Total execution time: {execution_time:.2f} seconds")
        return G, i, c_steps # Return the graph, iteration count, and clustering steps

    except TimeoutException:
        print("Function execution failed due to timeout.")
        execution_time = time.time() - start_time # Calculate execution time until timeout
        print(f"Total execution time: {execution_time:.2f} seconds")
        return G, i, c_steps  # Return the graph, iteration count, and clustering steps so far


# Model 2
def model2(G: nx.Graph, c_current: float, cluster: float, allowed_error: float, 
           nodes_per_round: int, seed: int = 42) -> Tuple[nx.Graph, int, List[float]]:
    """
    Modifies a graph to approximate a target clustering coefficient within a specified error margin,
    while ensuring connectivity and adjusting links dynamically based on extended neighborhood analysis.

    :param G: A networkx graph to be modified.
    :param c_current: Current clustering coefficient of the graph.
    :param cluster: Target clustering coefficient.
    :param allowed_error: Acceptable margin of error around the target clustering coefficient.
    :param nodes_per_round: Maximum number of new connections to be attempted in each iteration.
    :param seed: Seed for the random number generator for reproducibility.
    :return: A tuple containing the modified graph, the number of iterations completed, and
             a list capturing the history of clustering coefficients after each iteration.
    """
    random.seed(seed)  # Seed the random number generator for reproducibility
    c_steps = [c_current]  # Tracks changes of clustering coefficient
    i = 0  # Iteration counter
    cluster_bound = [cluster - allowed_error, cluster + allowed_error]  # Clustering bounds
    start_time = time.time()  # Record the start time

    # Setup timeout handler to avoid infinite execution
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3600)  # Set the alarm for 60 minutes

    try:
        while c_current < cluster_bound[0] or c_current > cluster_bound[1]:
            i += 1
            node_list = list(G.nodes())  # Refresh the node list each iteration

            # Ensure all nodes are part of the largest connected component
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                for node in node_list:
                    if node not in largest_cc:
                        G.add_edge(node, random.choice(list(largest_cc)))  # Connect isolated nodes to the largest component

            for node in node_list:
                if node not in G:
                    continue  # Skip iteration if node no longer exists

                current_links = list(G.neighbors(node)) # Get a list of links to the node's neighbors

                if current_links:
                    link_to_remove = random.choice(current_links) # Randomly select one link to remove
                    G.remove_edge(node, link_to_remove) # Remove the selected link

                # Gather neighbors of neighbors to form an extended neighbor set
                extended_neighbors = set(current_links)
                for neighbor in current_links:
                    extended_neighbors.update(G.neighbors(neighbor))
                extended_neighbors.discard(node)  # Remove the node itself from the set

                # Exclude current links and the node itself from potential new links
                possible_new_links = [n for n in node_list if n not in extended_neighbors]

                # Samples nodes for potential connection if there are more than enough candidates
                if len(possible_new_links) > nodes_per_round:
                    selected_nodes = random.sample(possible_new_links, nodes_per_round)
                else:
                    selected_nodes = possible_new_links # Use all possible links

                c_possible = []

                for possible in selected_nodes:
                    G.add_edge(node, possible) # Temporarily add the new link
                    c_possible.append(nx.average_clustering(G)) # Calculate new clustering coefficient
                    G.remove_edge(node, possible) # Remove the temporary link

                # Select the best new link if there are possible new links
                if c_possible:
                    best_node = selected_nodes[c_possible.index(max(c_possible))]
                    G.add_edge(node, best_node)

            c_current = nx.average_clustering(G)  # Update the clustering coefficient
            c_steps.append(c_current)  # Track the clustering coefficient progression

        signal.alarm(0)  # Disable the alarm
        execution_time = time.time() - start_time  # Calculate total execution time
        print(f"Total execution time: {execution_time:.2f} seconds")
        return G, i, c_steps # Return the graph, iteration count, and clustering steps

    except TimeoutException:
        print("Function execution failed due to timeout.")
        execution_time = time.time() - start_time  # Calculate execution time until timeout
        print(f"Total execution time: {execution_time:.2f} seconds")
        return G, i, c_steps  # Return the graph, iteration count, and clustering steps so far


# Model 3
def model3(G: nx.Graph, c_current: float, cluster: float, allowed_error: float,
           nodes_per_round: int, seed: int = 42) -> Tuple[nx.Graph, int, List[float]]:
    """
    Modifies a graph to approximate a target clustering coefficient within a specified error margin,
    attempting to rewire edges to nodes of similar degree.

    :param G: A networkx graph to be modified.
    :param c_current: Current clustering coefficient of the graph.
    :param cluster: Target clustering coefficient.
    :param allowed_error: Acceptable margin of error around the target clustering coefficient.
    :param nodes_per_round: Maximum number of new connections to attempt each round based on node degree similarity.
    :param seed: Seed for the random number generator for reproducibility.
    :return: A tuple containing the modified graph, the number of iterations completed, and
             a list capturing the history of clustering coefficients after each iteration.
    """
    random.seed(seed)  # Seed the random number generator for reproducibility
    c_steps = [c_current]  # Tracks changes of clustering coefficient
    i = 0  # Iteration counter
    cluster_bound = [cluster - allowed_error, cluster + allowed_error]  # Clustering bounds
    start_time = time.time()  # Record the start time

    # Setup timeout handler to avoid infinite execution
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3600)  # Set the alarm for 60 minutes

    try:
        while c_current < cluster_bound[0] or c_current > cluster_bound[1]:
            i += 1
            node_list = list(G.nodes())  # Refresh the node list each iteration

            # Ensure all nodes are part of the largest connected component
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                for node in node_list:
                    if node not in largest_cc:
                        G.add_edge(node, random.choice(list(largest_cc)))  # Connect isolated nodes to the largest component

            for node in node_list:
                if node not in G:
                    continue  # Skip iteration if node no longer exists

                current_links = list(G.neighbors(node)) # Get a list of links to the node's neighbors

                if current_links:
                    link_to_remove = random.choice(current_links) # Randomly select one link to remove
                    degree_of_removed_node = G.degree(link_to_remove)  # Degree of the node at the other end of the removed link
                    G.remove_edge(node, link_to_remove) # Remove the selected link

                # Filter potential new links to nodes with a degree matching that of the removed node
                possible_new_links = [n for n in node_list if G.degree(n) == degree_of_removed_node and n != node]

                # Samples nodes for potential connection if there are more than enough candidates
                if len(possible_new_links) > nodes_per_round:
                    selected_nodes = random.sample(possible_new_links, nodes_per_round)
                else:
                    selected_nodes = possible_new_links # Use all possible links

                c_possible = []

                for possible in selected_nodes:
                    G.add_edge(node, possible) # Temporarily add the new link
                    c_possible.append(nx.average_clustering(G)) # Calculate new clustering coefficient
                    G.remove_edge(node, possible) # Remove the temporary link

                # Select the best new link if there are possible new links
                if c_possible:
                    best_node = selected_nodes[c_possible.index(max(c_possible))]
                    G.add_edge(node, best_node)

            c_current = nx.average_clustering(G)  # Update the clustering coefficient
            c_steps.append(c_current)  # Track the clustering coefficient progression

        signal.alarm(0)  # Disable the alarm
        execution_time = time.time() - start_time  # Calculate total execution time
        print(f"Total execution time: {execution_time:.2f} seconds")
        return G, i, c_steps # Return the graph, iteration count, and clustering steps

    except TimeoutException:
        print("Function execution failed due to timeout.")
        execution_time = time.time() - start_time  # Calculate execution time until timeout
        print(f"Total execution time: {execution_time:.2f} seconds")
        return G, i, c_steps # Return the graph, iteration count, and clustering steps so far