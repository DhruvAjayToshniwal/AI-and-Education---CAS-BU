import openai
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

# Load data
df1 = pd.read_csv('BARD_Excel.csv')  # Replace with actual file path for File01
df2 = pd.read_csv('ChatGPT_Excel.csv')  # Replace with actual file path for File02

# Initialize OpenAI API client
openai.api_key = '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'  # Replace with actual OpenAI API key

def get_interconnected_triplets(question, answers):
    """
    Function to get interconnected triplets from OpenAI API, adjusted for 8 answers
    """
    system_message = f"Based on the following question and 8 answers:\n\nQuestion: '{question}'\n\n"
    for i, answer in enumerate(answers, start=1):
        system_message += f"Answer {i}: '{answer}'\n"
    system_message += "\nPlease create 8 interconnected triplets to represent this information in a knowledge graph. Format each triplet as 'Entity1:Relationship:Entity2'. The triplets should be interconnected, forming a cohesive structure when combined in a knowledge graph, with some nodes overlapping between the first and second sets of answers."
    
    messages = [{"role": "system", "content": system_message}]
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    
    return response.choices[0].message.content.strip().split('\n')

triplets_by_question = {}

# Assuming both files have the same questions in the same order
for index, row in df1.iterrows():
    question = row['Questions']
    answers_file1 = [row['Prompt1'], row['Prompt2'], row['Prompt3']]
    answers_file2 = df2.iloc[index][['Prompt1', 'Prompt2', 'Prompt3']].tolist()
    all_answers = answers_file1 + answers_file2  # Combining answers from both files for the same question
    triplets_by_question[question] = get_interconnected_triplets(question, all_answers)

def visualize_knowledge_graph(triplets_list, question, num):
    """
    Function to visualize knowledge graph
    """
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
    
    legend_info["lightblue"] = "First Set"
    legend_info["lightgreen"] = "Second Set"
    legend_info["red"] = "Common Node"
    
    plt.title(f"Knowledge Graph for: {question}", fontsize=20) 
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"Question_{num}_KG.png", format='PNG', bbox_inches='tight')
    plt.show()
    
    return legend_info

legend_by_question = {}

for count, (question, triplets) in enumerate(triplets_by_question.items(), start=1):
    legend_by_question[f"Question {count}"] = visualize_knowledge_graph(triplets, question, count)
    
for question, legend_info in legend_by_question.items():
    print(f"{question}:")
    for key, value in legend_info.items():
        print(f"  {key}: {value}")

# Load data
df = pd.read_csv('BARD_Excel.csv')  # Replace with your actual file path

# Initialize OpenAI API client
openai.api_key = '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' # Replace with actual OpenAI API key
mindmap_data_dict = {'Prompt1': [], 'Prompt2': [], 'Prompt3': []}

def get_mindmap_components(paragraph):
    """
    Function to get all components for a mind map from a paragraph using OpenAI API
    """
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

for index, row in df.iterrows():
    question = row['Questions']
    for prompt_key in ['Prompt1', 'Prompt2', 'Prompt3']:
        answer = row[prompt_key]
        mindmap_data_dict[prompt_key].append(get_mindmap_components(answer))

colors = {
    'Entities': 'skyblue',
    'Actions/Relations': 'lightgreen',
    'Attributes': 'lightcoral',
    'Concept Groups': 'violet',
    'Hierarchy': 'gold'
}

def parse_components(components_str):
    """
    Function to parse the structured response into components
    """
    components = {
        'Entities': [],
        'Actions/Relations': [],
        'Attributes': [],
        'Concept Groups': [],
        'Hierarchy': []
    }
    current_key = None
    valid_keys = components.keys()  

    for line in components_str.split('\n'):
        if ':' in line and any(key in line for key in valid_keys):
            current_key = line.split(':')[0]
        elif current_key in valid_keys and line.strip():
            item = line.strip().split('. ')[1] if '. ' in line else line.strip()
            components[current_key].append(item)

    return components

def create_and_display_mindmap(components_list, prompt_key):
    """
    Function to create and display a mind map
    """
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

    for components_str in components_list:
        components = parse_components(components_str)
        for component_type, items in components.items():
            all_components[component_type].extend(items)

    for component_type, items in all_components.items():
        for item in items:
            G.add_node(item, color=colors[component_type])

    for i, (component_type, items) in enumerate(all_components.items()):
        next_type_items = list(all_components.values())[(i + 1) % len(all_components)]
        for item in items:
            if next_type_items:
                G.add_edge(item, next_type_items[0])

    pos = nx.spring_layout(G, k=0.5, iterations=50)  
    nx.draw(G, pos, with_labels=True, node_size=500, 
            node_color=[G.nodes[n]['color'] for n in G.nodes],
            font_weight='bold', edge_color='gray', font_size=4)
    ax = plt.gca()
    ax.margins(0.1)

    plt.axis('off')
    plt.show()
    
for prompt_key, components_list in mindmap_data_dict.items():
    create_and_display_mindmap(components_list, prompt_key)

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

