import sys
from utilities import digest_network, analyze_graph
from models import model3

# This file demonstrates how to use the utilities.py and models.py files
# We will use the Eastern Massachusetts network and Model 3 as an example

# Change output_file to whatever file you'd like to output the results to
output_file = "example_output.txt"
stdout_backup = sys.stdout
sys.stdout = open(output_file, "w")

# Prior knowledge of the network being used is needed:
# We know EM has 74 nodes and c = 0.2869
network = "network_csvs/Eastern-Massachusetts.csv"
n = 74
c = 0.2869
print(f"n: {n}")
print(f"c: {c}")
with open(output_file, "a") as f:
    f.write(f"network: {network}\n")
    f.write(f"n: {n}\n")
    f.write(f"c: {c}\n")

# Digest the network csv to get the corresponding graph and average clustering coefficient
em_graph, em_c_avg = digest_network(network, n)
print(f"Average clustering coefficient of EM: {em_c_avg}")
with open(output_file, "a") as f:
    f.write(f"Average clustering coefficient of EM: {em_c_avg}\n")

# Use EM to generate a new network with Model 3
em_model, em_i, em_c_steps = model3(em_graph, em_c_avg, c, c * 0.1, 5)
print(f"Using {network} in Model 3 took {em_i} iterations to reach c = {em_c_steps[-1]}")
with open(output_file, "a") as f:
    f.write(f"Using {network} in Model 3 took {em_i} iterations to reach c = {em_c_steps[-1]}\n")

# Analyze the new network
# The resulting degree distribution plot is shown upon running
#   but it is not automatically saved, so please manually do so if you wish to keep it
analyze_graph(em_model)

sys.stdout.close()
sys.stdout = stdout_backup