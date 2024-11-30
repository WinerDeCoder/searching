import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React's URL   localhost
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

# ChromaDB setup with OpenAI embeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-large",
    api_key = OPENAI_API_KEY
)

# Initialize PersistentClient and create or retrieve collection
client = chromadb.PersistentClient(path="chromadb")
collection = client.get_or_create_collection(name="title_embed", embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})

# Define the model for the input (request body)
class SearchQuery(BaseModel):
    text: str

# Define the video search endpoint
@app.post("/api/title-search")
async def video_search(query: SearchQuery):
    search_text = query.text
    print("Text: ", search_text)
    try:
        # Perform ChromaDB query to get similar embeddings
        results = collection.query(
            query_texts=[search_text],
            n_results=7  # Return the most similar result (adjust as needed)
        )

        # Print results for debugging
        try:
            print(results, type(results))
            distances = results['distances'][0]
            documents = results['documents'][0]
            print("dis: ", distances)
            final_title = []
            for index in range(len(distances)):
                print("check index: ", distances[index])
                if distances[index] < 0.77:
                    final_title.append(documents[index])

            print("check title: ", final_title)
            return {"titles": final_title}
            # Extract video path from metadata if present
            
        except:
                return {"titles": []}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

