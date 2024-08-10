import pandas as pd
import numpy as np
import os
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


"""
getting classes from images
"""


# Initialize the YOLO model with the trained weights
model = YOLO('../runs/classify/train7/weights/last.pt')

# Define the directory containing test images
test_dir = '../components you have'

components_available = []

# Iterate through all images in the test directory
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(test_dir, filename)

        # Predict the image
        results = model(image_path, device='cuda')

        # Extract top-3 predictions
        names_list = [results[0].names[i] for i in results[0].probs.top5]
        prob_list = results[0].probs.top5conf.tolist()
        top_predictions = list(zip(names_list[:1], prob_list[:1]))

        # Print the image filename and predictions
        print(f"Image: {filename}")
        for name, prob in top_predictions:
            print(f"  {name}: {prob * 100:.2f}%")
            components_available.append(name)
        print()  # Add a newline for better readability between images

print(components_available)


"""
getting the relivant projects
"""

df = pd.read_csv('projects_dataset.csv')

"""
finding projects through comparison
"""

relevant_projects = []  # serial numbers of relevant projects
# components_available = ['7_segment_display_0.56_inch_white',  'arduino',  'battery',  'breadboard_power_supply']
relevant_columns = [df.columns.get_loc(i) for i in components_available]
print(relevant_columns)
for i in range(len(df)):
    # print(list(df.iloc[i]))
    for j in relevant_columns:
        if list(df.iloc[i])[j] == 1:
            print(df.columns[j], j, end='')
            relevant_projects.append(list(df.iloc[i])[0])
            print(list(df.iloc[i]))
            break

relevant_projects = sorted(list(set(relevant_projects)))
# print(relevant_projects)
for i in relevant_projects:
    print(list(df.iloc[i-1])[0], list(df.iloc[i-1])[1])


"""
getting prompt based recommendations using cosine similarity
"""

prompt = input('Describe your project needs in one sentence: ').lower()

for i in range(len(df)):
    df['summary']

summaries = df['summary']
vectorizer = TfidfVectorizer(stop_words='english')

#  to ensure that both the summaries and the prompt are represented in the same feature space:
all_texts = summaries.tolist() + [prompt]
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Split the TF-IDF matrix into summaries and prompt parts
summaries_tfidf = tfidf_matrix[:-1]
prompt_tfidf = tfidf_matrix[-1]

# Compute cosine similarity between the prompt and each summary
similarity_scores = cosine_similarity(prompt_tfidf, summaries_tfidf).flatten()

# Create a DataFrame with project serial numbers and their similarity scores
similarity_df = pd.DataFrame({
    'serial_number': df.index + 1,  # Assuming serial numbers start from 1
    'similarity_score': similarity_scores
})

# Sort the DataFrame by similarity score in descending order and get top 10
top_similar_projects = similarity_df.sort_values(by='similarity_score', ascending=False).head(10)

cosine_similar_projects = []

# Print the serial numbers of the top 10 most similar projects
print("Top 10 most similar projects:")
for index, row in top_similar_projects.iterrows():
    serial_number = int(row['serial_number'])
    print(f"Serial Number: {serial_number}, Similarity Score: {row['similarity_score']:.4f}")

    # Optionally, print the details of the top projects
    print(f"Project Details: {df.iloc[serial_number-1]['project_name']} - {df.iloc[serial_number-1]['summary']}")
    # print(index)
    cosine_similar_projects.append(index)

for i in cosine_similar_projects:
    print(list(df.iloc[i-1])[1])