import networkx as nx

def longest_simple_paths(graph, source, target):
    longest_paths = []
    longest_path_length = 0
    for path in nx.all_simple_paths(graph, source=source, target=target):
        if len(path) > longest_path_length:
            longest_path_length = len(path)
            longest_paths.clear()
            longest_paths.append(path)
        elif len(path) == longest_path_length:
            longest_paths.append(path)
    return longest_paths

tasks = nx.DiGraph()
tasks.add_edges_from([("src","encoding"),("encoding","framing"),("framing","lowAccuracy"),("lowAccuracy","inference"),("framing","highAccuracy"),("highAccuracy","inference"),("inference","packaging"),("highAccuracy","packaging"),("lowAccuracy","packaging"),("packaging","snk")])
for task in tasks.nodes():
    if (task=="src"):
        print(0)
    else:
        longest_paths = longest_simple_paths(tasks, source="src", target=task)
        print("from src to ", task," ",len(longest_paths[0]))