"""
app.py — Flask server
Routes:
  GET  /        → Landing page (index.html)
  GET  /chat    → Chat UI (chat.html)
  POST /get     → RAG chat endpoint
  GET  /health  → Health check
"""
import os, traceback
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from src.helper import get_embeddings
from src.prompt import get_prompt

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = os.environ.get("PINECONE_INDEX_NAME", "medibot")
LLM_BACKEND      = os.environ.get("LLM_BACKEND", "openai").lower()
MODEL_PATH       = os.environ.get("MODEL_PATH", "model/llama-2-7b-chat.ggmlv3.q4_0.bin")


def build_llm():
    if LLM_BACKEND == "openai":
        from langchain_community.chat_models import ChatOpenAI
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4,
                          openai_api_key=os.environ.get("OPENAI_API_KEY"))
    from langchain_community.llms import CTransformers
    return CTransformers(model=MODEL_PATH, model_type="llama",
                         config={"max_new_tokens": 512, "temperature": 0.7})


def build_chain():
    emb      = get_embeddings()
    Pinecone(api_key=PINECONE_API_KEY)
    store    = LangchainPinecone.from_existing_index(INDEX_NAME, emb)
    llm      = build_llm()
    prompt   = get_prompt()
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


qa = build_chain()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400
    try:
        result  = qa({"query": msg})
        answer  = result["result"].strip()
        sources = list(dict.fromkeys(
            d.metadata.get("source", "") for d in result.get("source_documents", [])
        ))
        return jsonify({"answer": answer, "sources": sources})
    except Exception:
        traceback.print_exc()
        return jsonify({"answer": "An internal error occurred. Please try again."}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "backend": LLM_BACKEND}), 200


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=8080, debug=debug)
