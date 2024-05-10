import sys 
import matplotlib.pyplot as plt
from utilities import digest_network, analyze_graph
from models import model3
import os


# This file demonstrates how to use the utilities.py and models.py files
# We will use the Eastern Massachusetts network and Model 3 as an example

# Prior knowledge of the network being used is needed:
# We know EM has 74 nodes and c = 0.2869
n = 74
c = 0.2869

output_file = "example_output.txt"
# If the file doesn't exist, create it
if not os.path.exists(output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
stdout_backup = sys.stdout
sys.stdout = open(output_file, "w")

# Digest the network csv to get the corresponding graph and average clustering coefficient
em_graph, em_c_avg = digest_network("network_csvs/Eastern-Massachusetts.csv", n)
print(f"Average clustering coefficient of EM: {em_c_avg}")
with open(output_file, "a") as f:
    f.write(f"Average clustering coefficient of EM: {em_c_avg}")

# Use EM to generate a new network with Model 3
em_model, em_i, em_c_steps = model3(em_graph, em_c_avg, c, c * 0.1, 5)
print(f"Using EM in Model 3 took {em_i} iterations to reach c = {em_c_steps[-1]}")
with open(output_file, "a") as f:
    f.write(f"Using EM in Model 3 took {em_i} iterations to reach c = {em_c_steps[-1]}\n")

# Analyze the new network
analyze_graph(em_model)

# Save the plot of degree distributions
plt.savefig("example_output_plot")

sys.stdout.close()
sys.stdout = stdout_backup