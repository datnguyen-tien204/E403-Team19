"""
rag/weaviate_store.py
Khởi tạo Weaviate vector store + helper để ingest tài liệu
"""
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import (
    GEMINI_API_KEY, GEMINI_EMBED_MODEL,
    WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_INDEX_NAME,
)


def get_weaviate_client() -> weaviate.WeaviateClient:
    """
    Kết nối Weaviate.
    - Local (Docker): WEAVIATE_URL=http://localhost:8080, không cần API key
    - Weaviate Cloud: WEAVIATE_URL=https://xxx.weaviate.network, cần API key
    """
    url = WEAVIATE_URL.rstrip("/")

    if WEAVIATE_API_KEY:
        # Weaviate Cloud Service
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )
    else:
        # Local Docker
        client = weaviate.connect_to_local(
            host=url.replace("http://", "").replace("https://", "").split(":")[0],
            port=int(url.split(":")[-1]) if ":" in url.split("//")[-1] else 8080,
        )

    return client


def get_vector_store(client: weaviate.WeaviateClient) -> WeaviateVectorStore:
    """Tạo LangChain WeaviateVectorStore với embedding local (HuggingFace)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # Model 420MB, hỗ trợ 50+ ngôn ngữ gồm tiếng Việt, chạy offline hoàn toàn
    )
    vector_store = WeaviateVectorStore(
        client=client,
        index_name=WEAVIATE_INDEX_NAME,
        text_key="text",
        embedding=embeddings,
    )
    return vector_store


def ingest_documents(vector_store: WeaviateVectorStore, docs: list[Document]) -> int:
    """
    Nạp danh sách Document vào Weaviate.
    Trả về số lượng doc đã thêm.
    """
    vector_store.add_documents(docs)
    return len(docs)


def ingest_text_files(vector_store: WeaviateVectorStore, file_paths: list[str]) -> int:
    """Helper: đọc file .txt và ingest vào vector store."""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    all_chunks = []

    for path in file_paths:
        loader = TextLoader(path, encoding="utf-8")
        raw = loader.load()
        chunks = splitter.split_documents(raw)
        # Gắn metadata nguồn
        for c in chunks:
            c.metadata["source"] = path
        all_chunks.extend(chunks)

    return ingest_documents(vector_store, all_chunks)