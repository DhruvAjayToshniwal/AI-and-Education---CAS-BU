import networkx as nx
import matplotlib.pyplot as plt
import openai

def parse_components(components_str):
    components = {
        'Entities': [],
        'Actions/Relations': [],
        'Attributes': [],
        'Concept Groups': [],
        'Hierarchy': []
    }
    current_key = None
    valid_keys = components.keys()  # Get a list of valid keys

    for line in components_str.split('\n'):
        # Check if the line introduces a new key and update current_key accordingly
        if ':' in line and any(key in line for key in valid_keys):
            current_key = line.split(':')[0]
        # Append items only if they belong to a recognized section
        elif current_key in valid_keys and line.strip():
            # Handling lines with or without numbering
            item = line.strip().split('. ')[1] if '. ' in line else line.strip()
            components[current_key].append(item)

    return components

def create_and_display_mindmap(components_list, prompt_key):
    G = nx.Graph()
    plt.figure(figsize=(12, 12))
    plt.title(f'Mind Map for {prompt_key}')

    all_components = {
        'Entities': [],
        'Actions/Relations': [],
        'Attributes': [],
        'Concept Groups': [],
        'Hierarchy': []
    }

    # Process each string in the components list
    for components_str in components_list:
        components = parse_components(components_str)
        for component_type, items in components.items():
            all_components[component_type].extend(items)

    # Add nodes with color coding for each component
    for component_type, items in all_components.items():
        for item in items:
            G.add_node(item, color=colors[component_type])

    # Connect nodes across different component types
    # For simplicity, connect each node to at least one node of a different type
    # This example demonstrates a basic and arbitrary connection logic
    for i, (component_type, items) in enumerate(all_components.items()):
        next_type_items = list(all_components.values())[(i + 1) % len(all_components)]
        for item in items:
            if next_type_items:  # Ensure there is at least one item in the next component type to connect
                # Connect each item to the first item of the next component type
                # This is an arbitrary choice to demonstrate cross-type connections
                G.add_edge(item, next_type_items[0])

    # Draw the graph with adjusted node size, font size, and layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjust 'k' for more space
    nx.draw(G, pos, with_labels=True, node_size=500, 
            node_color=[G.nodes[n]['color'] for n in G.nodes],
            font_weight='bold', edge_color='gray', font_size=4)
    ax = plt.gca()
    ax.margins(0.1)  # Add margins to ensure labels fit

    plt.axis('off')
    plt.show()
    