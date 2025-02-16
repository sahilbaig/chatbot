from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

cv_data = [
    "I have a Bachelor's degree in Computer Science.",
    "I worked as a Software Engineer at XYZ Corp from 2020 to 2022.",
    "My skills include Python, Machine Learning, and Chatbot Development.",
    "I completed a project on building a recommendation system using Python.",
    "I am proficient in data analysis and visualization using Pandas and Matplotlib."
]


cv_embeddings = model.encode(cv_data)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_response(query):
    # Encode the user query
    query_embedding = model.encode([query])
    
    # Compute cosine similarity between the query and CV data
    similarities = cosine_similarity(query_embedding, cv_embeddings)
    
    # Find the index of the most similar sentence
    best_match_index = np.argmax(similarities)
    
    # Return the best matching sentence from the CV data
    return cv_data[best_match_index]

queries = ["Hello"]
for query in queries:
    response = get_response(query)
    print(f"Query: {query}\nResponse: {response}\n")