import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import faiss
import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

app = FastAPI()


@app.get("/")
def serve_ui():
    return FileResponse("index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


class ChatRequest(BaseModel):
    user_id: str
    message: str


def ask_llm(question: str):
    response = llm.invoke(question)
    return response


@app.post("/chat")
def chat_endpoint(chat_req: ChatRequest):
    response = ask_llm(chat_req.message)
    return {"response": str(response.content)}


# --------Vector Database---------
dimension = 1536
index = faiss.IndexFlatL2(dimension)

user_memories = {}

embeddings = OpenAIEmbeddings()


def store_user_preference(user_id: str, text: str):
    vector = embeddings.embed_query(text)
    index.add(np.array([vector]))
    user_memories[user_id] = {"vector": vector, "text": text}


# ---------RAG----------
df = pd.read_csv("data/BGG_Data_Set.csv")


def retrieve_board_game_info(query: str):
    results = df[df["Name"].str.contains(query, case=False, na=False) |
                 df["Year Published"].str.contains(query, case=False, na=False) |
                 df["Min Players"].str.contains(query, case=False, na=False) |
                 df["Max Players"].str.contains(query, case=False, na=False) |
                 df["Play Time"].str.contains(query, case=False, na=False) |
                 df["Min Age"].str.contains(query, case=False, na=False) |
                 df["Users Rated"].str.contains(query, case=False, na=False) |
                 df["Rating Average"].str.contains(query, case=False, na=False) |
                 df["BGG Rank"].str.contains(query, case=False, na=False) |
                 df["Complexity Average"].str.contains(query, case=False, na=False) |
                 df["Owned Users"].str.contains(query, case=False, na=False) |
                 df["Mechanics"].str.contains(query, case=False, na=False) |
                 df["Domains"].str.contains(query, case=False, na=False)]

    return results.head(5).to_dict(orient="records")


@app.post("/search-game")
def search_game(chat_req: ChatRequest):
    games = retrieve_board_game_info(chat_req.message)
    return {"games": games}


@app.post("/recommend")
def recommend_game(chat_req: ChatRequest):
    user_id = chat_req.user_id
    if user_id in user_memories:
        user_prefs = user_memories[user_id]["text"]
        response = llm.predict(f"Recommend 5 board games based on these preferences: {user_prefs}")
        return {"recommendations": response}
    else:
        return {"message": "No preferences found. Start by sharing your favorite game genres or mechanics!"}
