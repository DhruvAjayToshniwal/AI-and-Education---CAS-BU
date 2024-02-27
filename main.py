import openai
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from knowledge_graph import visualize_knowledge_graph
from mind_map import create_and_display_mindmap

# Load data
df1 = pd.read_csv('BARD_Excel.csv')  # Replace with actual file path for File01
df2 = pd.read_csv('ChatGPT_Excel.csv')  # Replace with actual file path for File02

# Initialize OpenAI API client
openai.api_key = ''  # Replace with actual OpenAI API key

def get_interconnected_triplets(question, answers):
    """
    Function to get interconnected triplets from OpenAI API, adjusted for 8 answers
    """
    system_message = f"Based on the following question and 6 answers:\n\nQuestion: '{question}'\n\n"
    for i, answer in enumerate(answers, start=1):
        system_message += f"Answer {i}: '{answer}'\n"
    system_message += "\nPlease create 6 interconnected triplets to represent this information in a knowledge graph. Format each triplet as 'Entity1:Relationship:Entity2'. The triplets should be interconnected, forming a cohesive structure when combined in a knowledge graph, with some nodes overlapping between the first and second sets of answers."
    
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

legend_by_question = {}

for count, (question, triplets) in enumerate(triplets_by_question.items(), start=1):
    legend_by_question[f"Question {count}"] = visualize_knowledge_graph(triplets, question, count)
    
for question, legend_info in legend_by_question.items():
    print(f"{question}:")
    for key, value in legend_info.items():
        print(f"  {key}: {value}")

# Load data
df = pd.read_csv('BARD_Excel.csv')  # Replace with your actual file path

mindmap_data_dict = {'Prompt1': [], 'Prompt2': [], 'Prompt3': []}

for index, row in df.iterrows():
    question = row['Questions']
    for prompt_key in ['Prompt1', 'Prompt2', 'Prompt3']:
        answer = row[prompt_key]
        mindmap_data_dict[prompt_key].append(get_mindmap_components(answer))

for prompt_key, components_list in mindmap_data_dict.items():
    create_and_display_mindmap(components_list, prompt_key)