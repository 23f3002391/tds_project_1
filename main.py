import json
import pickle
import requests
import faiss
import numpy as np
from datetime import datetime
from typing import List
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Constants
AIPROXY_TOKEN =  "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDIzOTFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.-F2wmo2hdvNYDfY32Wg32VJdX3Lptd0f9LBsBgzwd_0"  # Replace with your actual token
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
base_url = "https://discourse.onlinedegree.iitm.ac.in"
start_date = datetime.fromisoformat('2025-01-01T00:00:00.000Z'.replace('Z', '+00:00'))
end_date = datetime.fromisoformat('2025-04-14T00:00:00.000Z'.replace('Z', '+00:00'))

cookies = {
    '_t': 'qslISl7fdmpoL%2BErBzp4WfGKPmow%2FGr0b6p%2FuvCay8zDmr8KLHZLGnUV16EPIyA4Q0i9aixVqEoaB4G5kwQ9M%2BfUJWdR1E%2BsnKEyoCDeVjXmorRJZH5zwR67zZAltoEmVvsw29dGF5toJ93UD7--AagGsOw1DaMvSmCX--08HFe%2FpW19jYktkso1fRfw%3D%3D',
    '_forum_session': 'b7Zx0I8qtfJXZqS5S3CGbpSqqge76X9Bjeew4Ph29uAhxHWiwq416Q%3D%3D--EHLrieYePUFVodl%2B--3tw8ybCVUnFX0BC7RSwkwA%3D%3D'
}

def load_faiss_index(index_path: str, metadata_path: str):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def embed_query(query: str, model):
    return model.encode([query])[0]

def search_top_k(query_vector: np.ndarray, index, metadata, k=5):
    query_vector = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query_vector, k)
    return [metadata[i] for i in indices[0] if i < len(metadata)]

def format_context(chunks: List[dict]) -> str:
    return "\n\n---\n\n".join([f"Source: {chunk['url']}\n{chunk['chunk']}" for chunk in chunks])

def get_ai_proxy_response(question: str, context: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a factual and helpful assistant. Your task is to answer the user's question strictly based on the provided context (i.e., the text chunks retrieved from the document index).\n\n"
                "Important instructions:\n"
                "- Do NOT answer based on prior knowledge or assumptions.\n"
                "- If the context does not provide enough information to answer the question, respond with: \"I don't have enough information to answer this question.\"\n"
                "- Only use statements from the provided chunks that are declarative (not questions or interrogative sentences).\n"
                "- For each piece of evidence you use, include the original URL and the *exact text excerpt* from that chunk that supports your answer.\n\n"
                "Your response must follow this exact JSON format:\n"
                '''{
                  "answer": "...",
                  "links": ["
                    {"
                      "url": "<URL of the chunk>","
                      "text": "<Exact supporting text from the chunk>"
                    },"
                    ..."
                  ],"
                  "summary": "<One-line summary of the answer>"
                "}'''
                "Ensure the URLs and texts you provide are actually referenced and directly support your answer. If nothing in the context supports an answer, say so clearly."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
    }
    response = requests.post(AIPROXY_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to get response - {response.status_code}", "details": response.text}

def scrape_conversation(topic_url):
    headers = { "User-Agent": "Mozilla/5.0" }
    try:
        topic_path = topic_url.split("/t/")[1].strip("/")
        topic_parts = topic_path.split("/")
        if len(topic_parts) == 2:
            topic_slug, topic_id = topic_parts
            topic_id = int(topic_id)
            extra_path = None
        elif len(topic_parts) == 3:
            topic_slug, topic_id, extra_path = topic_parts
            topic_id = int(topic_id)
        else:
            return None
    except:
        return None

    all_posts, page = [], 1
    while True:
        url = f"{base_url}/t/{topic_slug}/{topic_id}.json?page={page}"
        response = requests.get(url, headers=headers, cookies=cookies)
        if response.status_code != 200: break
        posts = response.json().get("post_stream", {}).get("posts", [])
        if not posts: break
        all_posts.extend(posts)
        page += 1

    if not all_posts: return None

    processed_posts = []
    for post in all_posts:
        soup = BeautifulSoup(post.get("cooked", ""), "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        if paragraphs:
            post_text = "\n".join(paragraphs)
            processed_posts.append(post_text)

    final_url = f"{base_url}/t/{topic_slug}/{topic_id}"
    if 'extra_path' in locals():
        final_url += f"/{extra_path}"

    return { "url": final_url, "posts": processed_posts }

def ask_question(query: str, specific_url: str = None):
    index, metadata = load_faiss_index("conversations1.index", "conversations_metadata1.pkl")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if specific_url:
        url_chunks = [m for m in metadata if m.get('url') == specific_url]
        if not url_chunks:
            convo = scrape_conversation(specific_url)
            if not convo: return { "error": "Could not scrape conversation." }
            new_chunks = []
            for post in convo['posts']:
                for para in post.split("\n"):
                    if para.strip():
                        vector = model.encode([para])[0]
                        new_chunks.append({ "url": convo['url'], "chunk": para, "vector": vector.tolist() })
            vectors = np.array([c['vector'] for c in new_chunks]).astype("float32")
            index.add(vectors)
            metadata.extend([{ "url": c['url'], "chunk": c['chunk'] } for c in new_chunks])
            faiss.write_index(index, "conversations1.index")
            with open("conversations_metadata1.pkl", "wb") as f:
                pickle.dump(metadata, f)
            try:
                with open("all_conversations1.json", "r+", encoding="utf-8") as f:
                    all_convos = json.load(f)
                    all_convos.append(convo)
                    f.seek(0)
                    json.dump(all_convos, f, indent=2, ensure_ascii=False)
            except: pass
        query_vec = embed_query(query, model)
        top_chunks = search_top_k(query_vec, index, metadata, k=200)
    else:
        query_vec = embed_query(query, model)
        top_chunks = search_top_k(query_vec, index, metadata, k=200)

    context = format_context(top_chunks)
    response = get_ai_proxy_response(query, context)
    return response


# 🔹 Vercel Handler
# def handler(request):
#     try:
#         body = request.get_json()
#         query = body.get("question")
#         url = request.get("link", None)

#         result = ask_question(query, url)
#         return {
#             "statusCode": 200,
#             "body": json.dumps(result)
#         }
#     except Exception as e:
#         return {
#             "statusCode": 500,
#             "body": json.dumps({ "error": str(e) })
#         }

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class QuestionPayload(BaseModel):
    question: str
    link: str = None  # Optional field

class Link(BaseModel):
    text: str
    url: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]  

# @app.post("/")
# async def handle_question(payload: QuestionPayload):
#     try:
#         # Call your model function (e.g., OpenAI)
#         result = ask_question(payload.question, payload.link)
        
#         # Safely parse content string to dictionary
#         content_str = result['choices'][0]['message']['content']
#         content_dict = json.loads(content_str)

#         # If you want to validate output structure
#         return content_dict

#     except Exception as e:
#         return {"error": str(e)}
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def handle_question(request: Request):
    try:
        # Call your OpenAI logic
    #     payload = await request.json()

    # # your logic here...
    #     result = ask_question(payload["question"], payload["link"])

    # # Step 1: parse the string to a real Python dict
    #     parsed = json.loads(result["choices"][0]["message"]["content"])
     # Step 1: Read the raw body as a string
        raw_body = await request.body()
        payload = json.loads(raw_body)  # Convert string to dict

        # Step 2: Call your question logic
        result = ask_question(payload["question"], payload.get("link", None))

        # Step 3: Parse OpenAI response content (which is itself a JSON string)
        content_str = result["choices"][0]["message"]["content"]
        # Step 4: Try to parse the content string as JSON
        try:
            parsed = json.loads(content_str)  # Try to parse as JSON
        except json.JSONDecodeError:
                parsed = {
        
        "answer": content_str.strip(),
        'links' : []
    }
            # parsed = json.loads(parsed)

        print(parsed)
        return parsed


        # Step 2: return proper structured JSON
        
        
        return parsed

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

# Local testing block
# if __name__ == "__main__":
    
#     from flask import Flask, request, jsonify

#     app = Flask(__name__)

#     @app.route("/api", methods=["POST","GET"])
#     def api():
#         if request.method=="POST":
#             try:
                
#                 return result["body"], result["statusCode"], {"Content-Type": "application/json"}
#             except Exception as e:
#                 return jsonify({ "error": str(e) }), 500

#     app.run(host="0.0.0.0", port=5000, debug=True)
