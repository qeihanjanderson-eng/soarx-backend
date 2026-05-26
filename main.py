# redeploy trigger
import glob
import os
import re
import shutil
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import openpyxl
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from services.fx_service import get_fx_rates
from services.market_service import get_market_snapshot

# ---------------------------
# LOGGING
# ---------------------------
logging.basicConfig(level=logging.INFO)

# ---------------------------
# OPTIONAL QDRANT IMPORTS
# ---------------------------
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

# ---------------------------
# CONSTANTS
# ---------------------------
BASE_DIR = os.path.dirname(__file__)

ENGINES_DIR = os.path.join(BASE_DIR, "engines")
CLIENT_FILES_DIR = os.path.join(BASE_DIR, "client_files")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

QDRANT_COLLECTION_NAME = "finance_engine"

EXPECTED_EXCEL_FILES = [
    "Master Excel Proto.xlsx",
    "Personal Finance Intelligence Engine.xlsx",
    "Pricing Model Valuation.xlsx",
]

# ---------------------------
# FASTAPI
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# ENV VARIABLES
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

openai_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

FRED_BASE_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)

# ---------------------------
# EMBEDDING MODEL
# ---------------------------
_embedder = None


def get_embedder():
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return _embedder


# ---------------------------
# HELPERS
# ---------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@app.on_event("startup")
def startup_event():
    ensure_dir(ENGINES_DIR)
    ensure_dir(CLIENT_FILES_DIR)
    ensure_dir(EXPORTS_DIR)

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is required"
        )

    if not FRED_API_KEY:
        print("WARNING: FRED_API_KEY missing")


# ---------------------------
# FRED HELPERS
# ---------------------------
def get_fred_latest(series_id: str):
    if not FRED_API_KEY:
        return None

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }

    try:
        resp = requests.get(
            FRED_BASE_URL,
            params=params,
            timeout=10,
        )

        data = resp.json()

        observations = data.get(
            "observations",
            [],
        )

        if not observations:
            return None

        value = observations[0]["value"]

        if value == ".":
            return None

        return float(value)

    except Exception as e:
        print(f"FRED error: {e}")
        return None


def get_risk_free_rate():
    return get_fred_latest("DGS3MO")


def get_inflation():
    return get_fred_latest("CPIAUCSL")


def get_fed_funds():
    return get_fred_latest("FEDFUNDS")


# ---------------------------
# YFINANCE HELPERS
# ---------------------------
def get_ticker_snapshot(ticker: str):
    try:
        t = yf.Ticker(ticker)

        try:
            info = t.get_info()
        except Exception:
            info = {}

        hist = t.history(period="1y")

        price = None

        if not hist.empty:
            price = float(
                hist["Close"].iloc[-1]
            )

        return {
            "ticker": ticker,
            "price": price,
            "market_cap": info.get("marketCap"),
            "revenue": info.get("totalRevenue"),
            "ebitda": info.get("ebitda"),
            "beta": info.get("beta"),
            "currency": info.get("currency"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }

    except Exception as e:
        print(f"Ticker snapshot error: {e}")
        return None


def get_fx_rate(pair: str):
    try:
        t = yf.Ticker(pair)

        hist = t.history(period="5d")

        if hist.empty:
            return None

        return float(
            hist["Close"].iloc[-1]
        )

    except Exception as e:
        print(f"FX error: {e}")
        return None


# ---------------------------
# EXCEL HELPERS
# ---------------------------
def read_excel_file(path: str) -> Dict[str, str]:
    workbook = openpyxl.load_workbook(
        path,
        data_only=False,
    )

    sheet_texts: Dict[str, str] = {}

    for sheet in workbook.worksheets:
        lines: List[str] = [
            f"Sheet: {sheet.title}"
        ]

        for row in sheet.iter_rows(
            values_only=False
        ):
            for cell in row:
                if cell.value is None:
                    continue

                if cell.data_type == "f":
                    lines.append(
                        f"{cell.coordinate}: formula={cell.value}"
                    )
                else:
                    lines.append(
                        f"{cell.coordinate}: {cell.value}"
                    )

        sheet_texts[sheet.title] = "\n".join(lines)

    return sheet_texts


def chunk_text(
    text: str,
    chunk_size: int = 400,
) -> List[str]:
    tokens = text.split()

    if not tokens:
        return []

    chunks = []

    for i in range(
        0,
        len(tokens),
        chunk_size,
    ):
        chunks.append(
            " ".join(tokens[i:i + chunk_size])
        )

    return chunks


# ---------------------------
# QDRANT
# ---------------------------
def ensure_qdrant_collection(client):
    existing = client.get_collections().collections

    names = [c.name for c in existing]

    if QDRANT_COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )


def embed_and_store(
    chunks: List[str],
    metadata: Dict[str, str],
) -> int:
    if not chunks:
        return 0

    from qdrant_client_loader import get_qdrant

    client = get_qdrant()

    ensure_qdrant_collection(client)

    vectors = get_embedder().encode(chunks)

    points = []

    for chunk, vector in zip(chunks, vectors):
        payload = {
            "text": chunk,
            **metadata,
        }

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=(
                    vector.tolist()
                    if hasattr(vector, "tolist")
                    else list(vector)
                ),
                payload=payload,
            )
        )

    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points,
    )

    return len(points)


def get_search_results(
    instruction: str,
    source_filter: Optional[str] = None,
    limit: int = 5,
):
    from qdrant_client_loader import get_qdrant

    client = get_qdrant()

    ensure_qdrant_collection(client)

    query_vector = (
        get_embedder()
        .encode(instruction)
        .tolist()
    )

    response = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    results = []

    for point in response.points:
        payload = point.payload or {}

        if (
            source_filter
            and payload.get("source")
            != source_filter
        ):
            continue

        results.append(payload)

    return results


# ---------------------------
# FILE HELPERS
# ---------------------------
def save_uploaded_file(
    upload_file: UploadFile,
    client_folder: str,
) -> str:
    ensure_dir(client_folder)

    safe_name = os.path.basename(
        upload_file.filename
    )

    destination = os.path.join(
        client_folder,
        safe_name,
    )

    with open(destination, "wb") as out_file:
        shutil.copyfileobj(
            upload_file.file,
            out_file,
        )

    return destination


def extract_text_from_pdf(path: str) -> str:
    text_lines: List[str] = []

    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text_lines.append(
                    page.extract_text() or ""
                )

        return "\n".join(text_lines)

    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


def extract_uploaded_file_texts(
    path: str,
) -> Dict[str, str]:
    _, extension = os.path.splitext(path)

    extension = extension.lower()

    if extension == ".xlsx":
        return read_excel_file(path)

    if extension == ".pdf":
        return {
            "PDF": extract_text_from_pdf(path)
        }

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


# ---------------------------
# WRITE TO EXCEL
# ---------------------------
def write_to_sheet(
    workbook,
    sheet_name,
    field_name,
    value,
):
    sheet = workbook[sheet_name]

    for row in sheet.iter_rows():
        for cell in row:
            if cell.value == field_name:
                target = sheet.cell(
                    row=cell.row,
                    column=cell.column + 1,
                )

                target.value = value
                return

    raise ValueError(
        f"Field not found: {field_name}"
    )


# ---------------------------
# INGEST FILES
# ---------------------------
def ingest_client_file(
    path: str,
    filename: str,
    client_folder: str,
) -> int:
    texts = extract_uploaded_file_texts(path)

    total_chunks = 0

    for sheet_name, text in texts.items():
        chunks = chunk_text(text)

        total_chunks += len(chunks)

        embed_and_store(
            chunks,
            {
                "source": "client_upload",
                "filename": filename,
                "sheet": sheet_name,
                "client_folder": client_folder,
            },
        )

    return total_chunks


# ---------------------------
# REQUEST MODELS
# ---------------------------
class ReasonRequest(BaseModel):
    query: str


# ---------------------------
# ROUTES
# ---------------------------
@app.get("/")
def healthcheck():
    return {
        "status": "ok"
    }


@app.post("/upload")
async def upload_file(
    client_id: str,
    file: UploadFile = File(...),
):
    try:
        client_folder = os.path.join(
            CLIENT_FILES_DIR,
            client_id,
        )

        path = save_uploaded_file(
            file,
            client_folder,
        )

        chunks = ingest_client_file(
            path,
            file.filename,
            client_id,
        )

        return {
            "success": True,
            "filename": file.filename,
            "chunks": chunks,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.get("/market")
def market_data():
    return {
        "risk_free_rate": get_risk_free_rate(),
        "inflation": get_inflation(),
        "fed_funds": get_fed_funds(),
    }


@app.get("/ticker/{ticker}")
def ticker_snapshot(ticker: str):
    data = get_ticker_snapshot(ticker)

    if not data:
        raise HTTPException(
            status_code=404,
            detail="Ticker not found",
        )

    return data


@app.get("/search")
def semantic_search(query: str):
    return {
        "results": get_search_results(query)
    }


@app.post("/reason")
def reason_endpoint(payload: ReasonRequest):
    instruction = payload.query.strip()

    if not instruction:
        raise HTTPException(
            status_code=400,
            detail="Query is required",
        )

    # ---------------------------
    # Rate limit guard
    # ---------------------------
    if len(instruction) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Query exceeds 2000 character limit",
        )

    logging.info(f"Incoming query: {instruction}")

    # ---------------------------
    # Semantic search
    # ---------------------------
    try:
        memory_results = get_search_results(
            instruction,
            limit=5,
        )

    except Exception as e:
        print(f"Memory search error: {e}")
        memory_results = []

    memory_text = "\n\n".join(
        [
            r.get("text", "")
            for r in memory_results
            if r.get("text")
        ]
    )

    # ---------------------------
    # Detect ticker + FX
    # ---------------------------
    detected_ticker = None
    detected_fx = None

    ticker_matches = re.findall(
        r"\b[A-Z]{1,5}\b",
        instruction.upper(),
    )

    fx_matches = re.findall(
        r"\b[A-Z]{3,6}=X\b",
        instruction.upper(),
    )

    if ticker_matches:
        detected_ticker = ticker_matches[0]

    if fx_matches:
        detected_fx = fx_matches[0]

    if (
        detected_fx
        and detected_fx == detected_ticker
    ):
        detected_ticker = None

    logging.info(
        f"Detected ticker: {detected_ticker}"
    )

    logging.info(
        f"Detected FX pair: {detected_fx}"
    )

    # ---------------------------
    # Market data
    # ---------------------------
    ticker_data = None
    fx_rate = None

    try:
        if detected_ticker:
            ticker_data = get_ticker_snapshot(
                detected_ticker
            )

    except Exception as e:
        print(f"Ticker fetch error: {e}")

    try:
        if detected_fx:
            fx_rate = get_fx_rate(
                detected_fx
            )

    except Exception as e:
        print(f"FX fetch error: {e}")

    # ---------------------------
    # Fallbacks
    # ---------------------------
    if ticker_data is None:
        ticker_data = {
            "error": "Ticker lookup failed"
        }

    if fx_rate is None:
        fx_rate = "Unavailable"

    # ---------------------------
    # Macro data
    # ---------------------------
    macro_data = {
        "risk_free_rate": get_risk_free_rate(),
        "inflation": get_inflation(),
        "fed_funds": get_fed_funds(),
    }

    logging.info(
        f"Macro data: {macro_data}"
    )

    # ---------------------------
    # Context
    # ---------------------------
    context = {
        "semantic_memory": memory_results[:3],
        "ticker_data": ticker_data,
        "fx_rate": fx_rate,
        "macro_data": macro_data,
    }

    if not openai_client:
        raise HTTPException(
            status_code=500,
            detail="OpenAI client not initialized",
        )

    # ---------------------------
    # Prompts
    # ---------------------------
    system_prompt = """
You are SoarX Financial Intelligence Engine.

You combine:
- Market data
- FX rates
- Macro data
- Semantic memory

Rules:
- Use ONLY supplied context
- Do not invent numbers
- If data is missing, explicitly state it
- Be concise, analytical, and structured
- Prefer bullet points for analysis
"""

    user_prompt = f"""
USER QUERY:
{instruction}

SEMANTIC MEMORY:
{memory_text}

CONTEXT:
{context}
"""

    # ---------------------------
    # OpenAI call
    # ---------------------------
    try:
        completion = (
            openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,
                max_tokens=800,
            )
        )

        answer = (
            completion
            .choices[0]
            .message
            .content
        )

    except Exception as e:
        print(f"OpenAI error: {e}")

        raise HTTPException(
            status_code=500,
            detail="LLM request failed",
        )

    # ---------------------------
    # Final response
    # ---------------------------
    return {
        "success": True,
        "query": instruction,
        "answer": answer,
        "ticker_used": detected_ticker,
        "fx_used": detected_fx,
        "macro_used": macro_data,
        "memory_used": memory_results[:3],
        "timestamp": (
            datetime.utcnow().isoformat()
        ),
    }


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
