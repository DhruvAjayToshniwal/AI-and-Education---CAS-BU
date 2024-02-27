#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import openai
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Load your data
df = pd.read_csv('File01.csv')  # Replace with your actual file path

# Initialize OpenAI API client
openai.api_key = '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'  # Replace with your actual OpenAI API key
triplets_dict = {'Prompt1': [], 'Prompt2': [], 'Prompt3': [], 'Prompt4': []}

# Function to get triplets from OpenAI API
def get_triplets(prompt, question, answer):
    messages = [
        {"role": "system", "content": f"For this {answer}\n\nPlease create a triplet for me to create knowledge graph which represents this. The format of the triplet should be in this style 'Entity1:Relationship:Entity2' for creating a knowledge graph. Only form the triplet and return it."}
    ]
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    print("Triplet")
    print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()
        
# Iterate over the dataframe and get triplets for each prompt
for index, row in df.iterrows():
    question = row['Questions']
    for prompt_key in ['Prompt1', 'Prompt2', 'Prompt3', 'Prompt4']:
        answer = row[prompt_key]
        triplets_dict[prompt_key].append(get_triplets(prompt_key, question, answer))


# In[1]:


import openai
import pandas as pd

# Assuming df1 and df2 are your DataFrames for File01 and File02, respectively
df1 = pd.read_csv('BARD_Excel.csv')  # Replace with your actual file path for File01
df2 = pd.read_csv('ChatGPT_Excel.csv')  # Replace with your actual file path for File02

# Initialize OpenAI API client
openai.api_key = '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'  # Replace with your actual OpenAI API key

# Function to get interconnected triplets from OpenAI API, adjusted for 8 answers
def get_interconnected_triplets(question, answers):
    system_message = f"Based on the following question and 8 answers:\n\nQuestion: '{question}'\n\n"
    for i, answer in enumerate(answers, start=1):
        system_message += f"Answer {i}: '{answer}'\n"
    system_message += "\nPlease create 8 interconnected triplets to represent this information in a knowledge graph. Format each triplet as 'Entity1:Relationship:Entity2'. The triplets should be interconnected, forming a cohesive structure when combined in a knowledge graph, with some nodes overlapping between the first and second sets of answers."
    
    messages = [{"role": "system", "content": system_message}]
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    
    #print("Interconnected Triplets:")
    #print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip().split('\n')

# Dictionary to store the triplets for each question
triplets_by_question = {}

# Assuming both files have the same questions in the same order
for index, row in df1.iterrows():
    question = row['Questions']
    answers_file1 = [row['Prompt1'], row['Prompt2'], row['Prompt3']]
    answers_file2 = df2.iloc[index][['Prompt1', 'Prompt2', 'Prompt3']].tolist()
    all_answers = answers_file1 + answers_file2  # Combining answers from both files for the same question
    triplets_by_question[question] = get_interconnected_triplets(question, all_answers)


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

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
    
    plt.title(f"Knowledge Graph for: {question}", fontsize=20)  # Increased title size
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"Question_{num}_KG.png", format='PNG', bbox_inches='tight')
    plt.show()
    
    return legend_info

legend_by_question = {}
#triplets_by_question = {}  # Make sure this variable is populated with your data
for count, (question, triplets) in enumerate(triplets_by_question.items(), start=1):
    legend_by_question[f"Question {count}"] = visualize_knowledge_graph(triplets, question, count)
    
# Print the legend information for later review
for question, legend_info in legend_by_question.items():
    print(f"{question}:")
    for key, value in legend_info.items():
        print(f"  {key}: {value}")


# In[3]:


def visualize_knowledge_graph(triplets_list, title, layout):
    G = nx.DiGraph()
    
    for triplet in triplets_list:
        parts = triplet.split(':')
        if len(parts) >= 3:
            entity1, entity2 = parts[0], parts[-1]
            relation = ':'.join(parts[1:-1])
            G.add_node(entity1)
            G.add_node(entity2)
            G.add_edge(entity1, entity2, label=relation)

    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.15, iterations=20)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("Invalid layout type")

    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
    
    plt.title(title)
    plt.axis('off')
    #plt.savefig(f"{file_name}_{layout}.png", format='PNG')  # Save before showing
    plt.show()
    plt.close()

# Example usage for different layouts
layouts = ['spring', 'circular', 'random', 'shell', 'kamada_kawai']
for layout in layouts:
    for question, triplets in triplets_by_question.items():
        visualize_knowledge_graph(triplets, question, layout)


# In[4]:


import openai
import pandas as pd

# Load your data
df = pd.read_csv('BARD_Excel.csv')  # Replace with your actual file path

# Initialize OpenAI API client
openai.api_key = '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' # Replace with your actual OpenAI API key
mindmap_data_dict = {'Prompt1': [], 'Prompt2': [], 'Prompt3': []}

# Function to get all components for a mind map from a paragraph using OpenAI API
def get_mindmap_components(paragraph):
    prompt = f"""Given the following information, create a concise list of components for a mind map. Keep descriptions brief and focus on clear, direct relationships:

{paragraph}

List the main subjects or nouns (Entities), actions or verbs (Actions/Relations), adjectives or descriptive phrases (Attributes), and suggest groupings (Concept Groups) and a hierarchy (Hierarchy) in a concise manner. Format your response as follows:
Entities:
Actions/Relations:
Attributes:
Concept Groups:
Hierarchy:"""

    messages = [{"role": "system", "content": prompt}]
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# Iterate over the dataframe and get mindmap components for each prompt
for index, row in df.iterrows():
    question = row['Questions']
    for prompt_key in ['Prompt1', 'Prompt2', 'Prompt3']:
        answer = row[prompt_key]
        # Call the function and store the response in a dictionary
        mindmap_data_dict[prompt_key].append(get_mindmap_components(answer))


# In[6]:


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Define colors for different components
colors = {
    'Entities': 'skyblue',
    'Actions/Relations': 'lightgreen',
    'Attributes': 'lightcoral',
    'Concept Groups': 'violet',
    'Hierarchy': 'gold'
}

# Function to parse the structured response into components
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


# Function to create and display a mind map
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
    
# Iterate over mindmap_data_dict to create and display mind maps for each prompt
for prompt_key, components_list in mindmap_data_dict.items():
    create_and_display_mindmap(components_list, prompt_key)


# In[ ]:


# Print the content of mindmap_data_dict to inspect its structure and data
for prompt_key, components_list in mindmap_data_dict.items():
    print(f"Prompt Key: {prompt_key}")
    for component in components_list:
        print(f"Component: {component}")
    print("\n")  # Add a newline for better readability between prompts


# In[ ]:





# In[ ]:


def visualize_knowledge_graph(triplets_list, title, file_name, layout):
    G = nx.DiGraph()
    
    for triplet in triplets_list:
        parts = triplet.split(':')
        if len(parts) >= 3:
            entity1, entity2 = parts[0], parts[-1]
            relation = ':'.join(parts[1:-1])
            G.add_node(entity1)
            G.add_node(entity2)
            G.add_edge(entity1, entity2, label=relation)

    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.15, iterations=20)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("Invalid layout type")

    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'label')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
    
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{file_name}_{layout}.png", format='PNG')  # Save before showing
    plt.show()
    plt.close()

# Example usage for different layouts
layouts = ['spring', 'circular', 'random', 'shell', 'kamada_kawai']
for layout in layouts:
    for prompt, triplets in triplets_dict.items():
        visualize_knowledge_graph(triplets, f"BARD Knowledge Graph for {prompt}", f"{prompt}", layout)


# In[ ]:


t


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset from CSV
df = pd.read_csv('ratingForBard.csv')  # Adjust the filename as needed

# Function to process the comma-separated string values in each cell
def process_column(column):
    # Split the string by commas, convert to integers, and flatten the list
    return [int(value) for value_list in column.dropna().str.split(',') for value in value_list]

# Apply the processing function to each column and create a new DataFrame with processed values
processed_data = pd.DataFrame({col: process_column(df[col]) for col in df.columns})

# Calculate the mean and median for each category in the processed data
mean_values = processed_data.mean()
median_values = processed_data.median()

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

colors = ['lightblue' if (x not in ['Creativity', 'Depth']) else 'red' for x in mean_values.index]

# Plot for mean values with highlight for Creativity and Depth
axes[0].bar(mean_values.index, mean_values.values, color=colors)
axes[0].set_title('Gemini Rating Mean Values')
axes[0].set_ylim(0, 6)
for i, value in enumerate(mean_values):
    axes[0].text(i, value + 0.1, f"{value:.2f}", ha='center')

# Plot for median values
axes[1].bar(median_values.index, median_values.values, color=colors)
axes[1].set_title('Gemini Rating Median Values')
axes[1].set_ylim(0, 6)
for i, value in enumerate(median_values):
    axes[1].text(i, value + 0.1, f"{value:.2f}", ha='center')
    
    # Highlighting text to draw attention
axes[0].annotate('Low Creativity and Depth', xy=(-0.5, mean_values['Creativity'] + 0.5),
                 xytext=(-0.5, mean_values['Creativity'] + 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=9, color='red')

axes[1].annotate('Low Creativity and Depth', xy=(-0.5, median_values['Creativity'] + 0.5),
                 xytext=(-0.5, median_values['Creativity'] + 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=9, color='red')

plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('ratingForBard.csv')

# Drop rows with NaN values
df = df.dropna()

# Define a function to convert comma-separated strings into a list of integers
def process_column(column):
    return [[int(rating) for rating in value.split(',')] for value in column]

# Process each column in the dataframe
categories = df.columns
ratings = {category: process_column(df[category]) for category in categories}

# Create a new dataframe for plotting
plot_data = {category: {'Prompt1': [], 'Prompt2': [], 'Prompt3': []} for category in categories}
for category in categories:
    for triplet in ratings[category]:
        for i, rating in enumerate(triplet):
            plot_data[category][f'Prompt{i+1}'].append(rating)

# Convert the processed plot data into a DataFrame for easier plotting
plot_df = pd.DataFrame({(i,j): plot_data[i][j] for i in plot_data.keys() for j in plot_data[i].keys()})

# Plotting
n_categories = len(categories)
n_prompts = 3
bar_width = 0.25
index = np.arange(n_categories)
fig, ax = plt.subplots()

colors = ['red', 'blue', 'green']  # Colors for each prompt

for i in range(n_prompts):
    prompt_ratings = [plot_df[(category, f'Prompt{i+1}')].mean() for category in categories]
    ax.bar(index + i*bar_width, prompt_ratings, bar_width, label=f'Prompt {i+1}', color=colors[i])

# Add some text for labels, title and axes ticks
ax.set_xlabel('Category')
ax.set_ylabel('Ratings')
ax.set_title('Ratings by category and prompt')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(categories)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)

plt.show()


# In[ ]:


# Let's first read the new CSV file and take a look at its structure.
df_new = pd.read_csv('ratingResponse.csv')

# Display the first few rows to understand the structure
df_new.head()

n_questions = len(df_new) // 3
reshaped_data = df_new.values.reshape((n_questions, 3, -1))

# Calculate the mean rating for each prompt per category for each question
mean_ratings_per_question = reshaped_data.mean(axis=0)

# Create a new DataFrame for the mean ratings per prompt across all questions
mean_ratings_df = pd.DataFrame(mean_ratings_per_question, columns=df_new.columns)

# Now we can plot the data
fig, ax = plt.subplots()

# Define the bar width and the indices for the groups
bar_width = 0.25
indices = np.arange(len(df_new.columns))

# Colors for each prompt as specified
colors = ['red', 'blue', 'green']  # Colors for each prompt

# Plotting the bars for each prompt
for i in range(3):
    ax.bar(indices + i * bar_width, mean_ratings_df.iloc[i], bar_width, label=f'Prompt {i+1}', color=colors[i])

# Add some text for labels, title and axes ticks
ax.set_xlabel('Category')
ax.set_ylabel('Average Ratings')
ax.set_title('Average Ratings by Category and Prompt')
ax.set_xticks(indices + bar_width)
ax.set_xticklabels(df_new.columns)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)

plt.show()


# In[ ]:


import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, mannwhitneyu

# Load and preprocess the original dataset
df_original = pd.read_csv('ratingForBard.csv')  # Adjust the path as necessary
df_original = df_original.dropna()
def process_column(column):
    return [[int(rating) for rating in value.split(',')] for value in column]

categories = df_original.columns
ratings_original = {category: process_column(df_original[category]) for category in categories}

# Transforming the original ratings into an array matching the new dataset's structure
data_original = np.array([values for values_list in ratings_original.values() for values in values_list]).reshape(-1, 3, len(categories))

# Load the new dataset
df_new = pd.read_csv('ratingResponse.csv')  # Adjust the path as necessary
data_new = df_new.values.reshape(-1, 3, len(categories))

# Ensure both datasets have the same number of observations
data_original_trimmed = data_original[:len(data_new), :, :]

# Perform statistical tests
t_test_results = {}
mannwhitneyu_results = {}

for i, category in enumerate(categories):
    t_test_results[category] = []
    mannwhitneyu_results[category] = []
    for j in range(3):
        t_stat, t_p = ttest_rel(data_original_trimmed[:, j, i], data_new[:, j, i])
        u_stat, u_p = mannwhitneyu(data_original_trimmed[:, j, i], data_new[:, j, i], alternative='two-sided')
        t_test_results[category].append(t_p)  # Store p-values for simplicity
        mannwhitneyu_results[category].append(u_p)

# Convert results to DataFrame for tabular representation
t_test_df = pd.DataFrame(t_test_results, index=['Prompt1', 'Prompt2', 'Prompt3'])
mannwhitneyu_df = pd.DataFrame(mannwhitneyu_results, index=['Prompt1', 'Prompt2', 'Prompt3'])

print("Student's t-test P-Values:")
print(t_test_df)
print("\nMann-Whitney U Test P-Values:")
print(mannwhitneyu_df)


# In[ ]:




