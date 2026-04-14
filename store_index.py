"""
store_index.py — Run ONCE to ingest PDFs into Pinecone.
Usage: python store_index.py
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from src.helper import load_pdf_documents, split_documents, get_embeddings

load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = os.environ.get("PINECONE_INDEX_NAME", "medibot")
EMBED_DIM        = 384  # all-MiniLM-L6-v2


def main():
    print("=" * 55)
    print("  MediBot — Document Ingestion Pipeline")
    print("=" * 55)

    docs   = load_pdf_documents("data/")
    if not docs:
        raise FileNotFoundError("No PDFs in ./data/ — add medical PDFs and retry.")

    chunks     = split_documents(docs)
    embeddings = get_embeddings()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"Creating index '{INDEX_NAME}'…")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    print(f"Upserting {len(chunks)} chunks…")
    LangchainPinecone.from_documents(chunks, embeddings, index_name=INDEX_NAME)
    print("✓ Ingestion complete!")


if __name__ == "__main__":
    main()
