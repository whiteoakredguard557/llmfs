"""
code_search.py — Ingest a codebase into LLMFS and run semantic search queries.

Demonstrates:
  • Ingesting multiple Python source files with ``content_type="python"``
  • Tagging memories by module / feature area
  • Semantic search queries that cross module boundaries
  • Filtering search results by tag or layer
  • Reading the full content of a specific result

No external API keys required.  Uses the local all-MiniLM-L6-v2 embedder.

Run:
    python examples/code_search.py
"""

from __future__ import annotations

import tempfile
import textwrap

from llmfs import MemoryFS


# ── Synthetic codebase snippets ────────────────────────────────────────────────
# In a real scenario you would walk the filesystem:
#   for p in Path("src").rglob("*.py"):
#       mem.write(f"/code{p}", content=p.read_text(), content_type="python", ...)

CODEBASE: list[dict] = [
    {
        "path": "/code/auth/jwt_handler.py",
        "tags": ["auth", "jwt", "security"],
        "content": textwrap.dedent("""\
            # auth/jwt_handler.py
            import jwt
            import time

            SECRET_KEY = "supersecret"
            JWT_EXPIRY_SECONDS = 3600

            def create_token(user_id: str) -> str:
                payload = {
                    "sub": user_id,
                    "iat": int(time.time()),
                    "exp": int(time.time()) + JWT_EXPIRY_SECONDS,
                }
                return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

            def verify_token(token: str) -> dict:
                \"\"\"Raises jwt.ExpiredSignatureError if token is expired.\"\"\"
                return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        """),
    },
    {
        "path": "/code/auth/oauth.py",
        "tags": ["auth", "oauth", "security"],
        "content": textwrap.dedent("""\
            # auth/oauth.py
            import requests

            GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

            def exchange_code(code: str, redirect_uri: str, client_id: str, client_secret: str) -> dict:
                \"\"\"Exchange an OAuth2 authorisation code for tokens.\"\"\"
                resp = requests.post(GOOGLE_TOKEN_URL, data={
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "authorization_code",
                })
                resp.raise_for_status()
                return resp.json()

            def refresh_access_token(refresh_token: str, client_id: str, client_secret: str) -> dict:
                \"\"\"Use a refresh token to get a new access token.\"\"\"
                resp = requests.post(GOOGLE_TOKEN_URL, data={
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "refresh_token",
                })
                resp.raise_for_status()
                return resp.json()
        """),
    },
    {
        "path": "/code/storage/chunker.py",
        "tags": ["storage", "chunking", "embeddings"],
        "content": textwrap.dedent("""\
            # storage/chunker.py
            \"\"\"Adaptive text chunker for code and prose.\"\"\"
            from __future__ import annotations
            import ast

            DEFAULT_CHUNK_SIZE = 512  # tokens

            class AdaptiveChunker:
                \"\"\"Splits text into semantically coherent chunks.

                For Python source: splits at function/class boundaries using AST.
                For prose: splits at paragraph / heading boundaries.
                \"\"\"

                def chunk(self, text: str, content_type: str | None = None) -> list[str]:
                    if content_type == "python":
                        return self._chunk_python(text)
                    return self._chunk_prose(text)

                def _chunk_python(self, source: str) -> list[str]:
                    try:
                        tree = ast.parse(source)
                    except SyntaxError:
                        return [source]
                    chunks = []
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            chunks.append(ast.get_source_segment(source, node) or "")
                    return chunks or [source]

                def _chunk_prose(self, text: str) -> list[str]:
                    paragraphs = [p.strip() for p in text.split("\\n\\n") if p.strip()]
                    return paragraphs or [text]
        """),
    },
    {
        "path": "/code/storage/vector_store.py",
        "tags": ["storage", "vector", "chromadb", "embeddings"],
        "content": textwrap.dedent("""\
            # storage/vector_store.py
            \"\"\"ChromaDB wrapper for semantic search.\"\"\"
            import chromadb

            class VectorStore:
                def __init__(self, path: str) -> None:
                    self._client = chromadb.PersistentClient(path=path)
                    self._col = self._client.get_or_create_collection("memories")

                def upsert(self, ids: list[str], embeddings: list[list[float]],
                           documents: list[str], metadatas: list[dict]) -> None:
                    self._col.upsert(ids=ids, embeddings=embeddings,
                                     documents=documents, metadatas=metadatas)

                def query(self, embedding: list[float], k: int = 5,
                          where: dict | None = None) -> list[dict]:
                    kwargs: dict = {"query_embeddings": [embedding], "n_results": k,
                                    "include": ["documents", "metadatas", "distances"]}
                    if where:
                        kwargs["where"] = where
                    return self._col.query(**kwargs)

                def delete(self, ids: list[str]) -> None:
                    self._col.delete(ids=ids)
        """),
    },
    {
        "path": "/code/api/endpoints.py",
        "tags": ["api", "http", "fastapi"],
        "content": textwrap.dedent("""\
            # api/endpoints.py
            from fastapi import FastAPI, Depends, HTTPException, status
            from fastapi.security import OAuth2PasswordBearer

            from auth.jwt_handler import verify_token

            app = FastAPI(title="MyApp API")
            oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

            async def get_current_user(token: str = Depends(oauth2_scheme)):
                try:
                    payload = verify_token(token)
                    return payload["sub"]
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired token",
                    )

            @app.get("/me")
            async def read_me(user_id: str = Depends(get_current_user)):
                return {"user_id": user_id}

            @app.post("/data")
            async def write_data(payload: dict, user_id: str = Depends(get_current_user)):
                # Process payload ...
                return {"status": "ok", "written_by": user_id}
        """),
    },
    {
        "path": "/code/utils/retry.py",
        "tags": ["utils", "reliability"],
        "content": textwrap.dedent("""\
            # utils/retry.py
            import time
            import functools
            from typing import Callable, TypeVar

            T = TypeVar("T")

            def with_retry(max_attempts: int = 3, backoff: float = 1.0):
                \"\"\"Decorator: retry on exception with exponential backoff.\"\"\"
                def decorator(fn: Callable[..., T]) -> Callable[..., T]:
                    @functools.wraps(fn)
                    def wrapper(*args, **kwargs) -> T:
                        for attempt in range(1, max_attempts + 1):
                            try:
                                return fn(*args, **kwargs)
                            except Exception as exc:
                                if attempt == max_attempts:
                                    raise
                                sleep = backoff * (2 ** (attempt - 1))
                                print(f"Attempt {attempt} failed: {exc}. Retrying in {sleep}s...")
                                time.sleep(sleep)
                    return wrapper
                return decorator
        """),
    },
]


# ── Ingest and query ───────────────────────────────────────────────────────────

def ingest_codebase(mem: MemoryFS) -> None:
    """Store every synthetic source file in LLMFS."""
    print("Ingesting codebase…")
    for entry in CODEBASE:
        obj = mem.write(
            entry["path"],
            content=entry["content"],
            layer="knowledge",
            tags=entry["tags"],
            content_type="python",   # ← tells the chunker to use AST-aware splitting
            source="code_search_demo",
        )
        print(f"  ✓ {entry['path']}  ({len(obj.chunks)} chunk(s))")


def run_queries(mem: MemoryFS) -> None:
    """Demonstrate several semantic search patterns."""

    queries = [
        # (description, query, kwargs)
        ("General: authentication",
         "authentication and token validation",
         {}),
        ("Specific: chunking / splitting text",
         "split text into smaller pieces for embedding",
         {}),
        ("Specific: vector store operations",
         "store and query embedding vectors",
         {}),
        ("Tag-filtered: security-related files only",
         "token expiry and refresh",
         {"tags": ["security"]}),
        ("Layer-filtered: knowledge layer only",
         "retry logic on failed requests",
         {"layer": "knowledge"}),
    ]

    for description, query, kwargs in queries:
        results = mem.search(query, k=3, **kwargs)
        print(f"\n── {description}")
        print(f"   Query: {query!r}")
        if not results:
            print("   (no results)")
            continue
        for r in results:
            snippet = r.chunk_text[:100].replace("\n", " ")
            print(f"   [{r.score:.3f}] {r.path}")
            print(f"            {snippet}…")


def read_top_result(mem: MemoryFS) -> None:
    """Show how to read the full content of the highest-scoring result."""
    results = mem.search("JWT token creation", k=1)
    if not results:
        return

    top = results[0]
    print(f"\n── Full read of top result: {top.path}")
    obj = mem.read(top.path, query="token creation expiry")
    print(obj.content)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="llmfs_codesearch_") as tmp:
        mem = MemoryFS(path=tmp)
        print(f"MemoryFS at: {tmp}\n")

        ingest_codebase(mem)
        print()
        run_queries(mem)
        read_top_result(mem)

        status = mem.status()
        print(f"\nTotal memories: {status['total']}, disk: {status['disk_mb']:.2f} MB")
