from fastapi import FastAPI
from test import greet
from index import get_response  # Ensure this is an async function

app = FastAPI()

@app.get("/data")
async def get_data():
    queries = ["What languages do you know?"]
    
    responses = []
    for query in queries:
        response = await get_response(query)  # Await the async function
        responses.append({"query": query, "response": response})

    return { "responses": responses}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
