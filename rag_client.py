import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for directories whose names include "chroma" (common persist dirs)
    for d in current_dir.rglob("*"):
        if not d.is_dir():
            continue
        if "chroma" not in d.name.lower():
            continue

        try:
            client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(d)))
            # list_collections returns metadata about collections
            collections = []
            try:
                collections = client.list_collections() or []
            except Exception:
                # older clients may not support list_collections; skip
                collections = []

            for col in collections:
                name = col.get("name") if isinstance(col, dict) else getattr(col, "name", None)
                if not name:
                    continue
                key = f"{d}/{name}"
                display = f"{name} @ {d.name}"
                # try to get counts; some implementations may not expose count API
                try:
                    c = client.get_collection(name)
                    count = getattr(c, "count", None)
                    if callable(count):
                        count = count()
                except Exception:
                    count = None

                backends[key] = {
                    "path": str(d),
                    "directory": str(d),
                    "collection": name,
                    "collection_name": name,
                    "display": display,
                    "display_name": display,
                    "count": count,
                }
        except Exception:
            # skip inaccessible directories
            continue

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_dir))
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(name=collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    filter_dict = None
    if mission_filter and mission_filter.lower() not in ("all", "none", "", "any"):
        # normalize mission filter to lowercase and underscores
        mf = mission_filter.lower().replace(' ', '_')
        filter_dict = {"mission": mf}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict,
            include=["metadatas", "documents", "distances"],
        )

        # chroma returns lists per-query: normalize and deduplicate documents by content
        docs = []
        metas = []
        dists = []

        if results is None:
            return None

        # support both dict and object-like responses
        documents = results.get('documents', [[]])[0] if isinstance(results, dict) else []
        metadatas = results.get('metadatas', [[]])[0] if isinstance(results, dict) else []
        distances = results.get('distances', [[]])[0] if isinstance(results, dict) else []

        seen = set()
        for i, doc in enumerate(documents):
            key = (doc or '').strip()[:500]
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            docs.append(doc)
            metas.append(metadatas[i] if i < len(metadatas) else {})
            dists.append(distances[i] if i < len(distances) else None)

        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}
    except Exception:
        # Try a fallback without 'where' if filter fails
        try:
            results = collection.query(query_texts=[query], n_results=n_results, include=["metadatas", "documents", "distances"])
            if results is None:
                return None
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            seen = set()
            docs, metas, dists = [], [], []
            for i, doc in enumerate(documents):
                key = (doc or '').strip()[:500]
                if not key or key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
                metas.append(metadatas[i] if i < len(metadatas) else {})
                dists.append(distances[i] if i < len(distances) else None)
            return {"documents": [docs], "metadatas": [metas], "distances": [dists]}
        except Exception:
            return None

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    parts = ["Retrieved context from documents:"]
    max_chunk_chars = 1500

    for i, doc in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) else {}
        mission = meta.get("mission", "unknown") if isinstance(meta, dict) else "unknown"
        mission = str(mission).replace("_", " ").title()
        source = meta.get("source", meta.get("filepath", "unknown")) if isinstance(meta, dict) else "unknown"
        category = meta.get("category", "") if isinstance(meta, dict) else ""
        if category:
            category = str(category).replace("_", " ").title()

        header = f"Source {i+1} | Mission: {mission} | Source: {source}"
        if category:
            header += f" | Category: {category}"

        parts.append(header)

        text = doc or ""
        if len(text) > max_chunk_chars:
            parts.append(text[:max_chunk_chars] + "... [truncated]")
        else:
            parts.append(text)

        parts.append("---")

    return "\n".join(parts)