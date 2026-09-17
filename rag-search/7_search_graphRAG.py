import networkx as nx

# 1. Construct Knowledge Graph
G = nx.Graph()
G.add_edge("Arun Paria", "LanceDB Project", relation="DEVELOPER_OF")
G.add_edge("LanceDB Project", "HuggingFace Model", relation="USES")

# 2. Graph Traversal Query: What models are connected to Arun Paria?
nodes_related = list(nx.single_source_shortest_path_length(G, "Arun Paria", cutoff=2).keys())

print(f"Entities related within 2 hops: {nodes_related}")
# Output: ['Arun Paria', 'LanceDB Project', 'HuggingFace Model']