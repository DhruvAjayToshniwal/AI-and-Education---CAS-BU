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