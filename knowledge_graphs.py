import networkx as nx
import matplotlib.pyplot as plt

def visualize_knowledge_graph(triplets_list, question, num):
    G = nx.DiGraph()
    color_map = []
    edge_color_map = []
    node_label_mapping = {}
    legend_info = {}
    node_counter = 1
    
    first_set = set()
    second_set = set()
    
    for i, triplet in enumerate(triplets_list):
        if triplet:
            clean_triplet = triplet.split(':', 1)[1].strip(" '")
            parts = clean_triplet.split(':')
            if len(parts) == 3:
                entity1, relation, entity2 = [part.strip() for part in parts]
                
                if entity1 not in node_label_mapping:
                    node_label_mapping[entity1] = str(node_counter)
                    legend_info[str(node_counter)] = entity1
                    node_counter += 1
                if entity2 not in node_label_mapping:
                    node_label_mapping[entity2] = str(node_counter)
                    legend_info[str(node_counter)] = entity2
                    node_counter += 1
                
                G.add_edge(entity1, entity2, label=relation)
                
                if i < 4:
                    first_set.add(entity1)
                    first_set.add(entity2)
                    edge_color_map.append('blue')
                else:
                    second_set.add(entity1)
                    second_set.add(entity2)
                    edge_color_map.append('green')
    
    common_nodes = first_set.intersection(second_set)
    
    for node in G.nodes():
        if node in common_nodes:
            color_map.append('red')
        elif node in first_set:
            color_map.append('lightblue')
        else:
            color_map.append('lightgreen')
    
    pos = nx.spring_layout(G, k=1.0, iterations=50)
    
    plt.figure(figsize=(16, 10))
    nx.draw(G, pos, node_color=color_map, edge_color=edge_color_map, with_labels=False, node_size=2700, width=4, arrowstyle='->', arrowsize=10)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=13)
    
    for node, (x, y) in pos.items():
        plt.text(x, y, node_label_mapping[node], fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Add color meanings to legend_info
    legend_info["lightblue"] = "First Set"
    legend_info["lightgreen"] = "Second Set"
    legend_info["red"] = "Common Node"
    
    plt.title(f"{question}", fontsize=17)  # Increased title size
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"Question_{num}_KG.png", format='PNG', bbox_inches='tight')
    plt.show()
    
    return legend_info