import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

cv_data = [
    "I have a Bachelor's degree in Computer Science.",
    "I worked as a Software Engineer at XYZ Corp from 2020 to 2022.",
    "My skills include Python, Machine Learning, and Chatbot Development.",
    "I completed a project on building a recommendation system using Python.",
    "I am proficient in data analysis and visualization using Pandas and Matplotlib."
]

# Precompute embeddings
cv_embeddings = model.encode(cv_data)

async def get_response(query):
    # Encode the query asynchronously
    query_embedding = await asyncio.to_thread(model.encode, [query])
    
    # Compute cosine similarity asynchronously
    similarities = await asyncio.to_thread(cosine_similarity, query_embedding, cv_embeddings)
    
    # Find the best match asynchronously
    best_match_index = await asyncio.to_thread(np.argmax, similarities)
    
    return cv_data[best_match_index]
