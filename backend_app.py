"""
FastAPI backend for PersonaRAGSystem.

Dependencies:
 pip install fastapi uvicorn sentence-transformers chromadb gpt4all numpy regex

Run:
 uvicorn persona_rag_backend:app --reload --port 8000
"""

import os
import re
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from rag_system import PersonaRAGSystem  # Your class file
from config import settings  # Your settings file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="Persona RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store PersonaRAGSystem instances in memory per persona_id
persona_systems: Dict[int, PersonaRAGSystem] = {}


# -----------------------
# Text cleaning function
# -----------------------
def clean_response_text(text: str) -> str:
    """
    Removes reasoning, hashtags, emojis, URLs, and unwanted symbols from model responses.
    Keeps only relevant human-readable answer text.
    """
    if not text:
        return ""

    # Remove URLs and handles
    text = re.sub(r"http\S+|pic\.twitter\.com\S+|@\S+", "", text)

    # Remove hashtags and emojis (non-ASCII)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Remove 'Reasoning:' and any explanation after it
    text = re.sub(r"Reasoning:.*", "", text, flags=re.DOTALL)

    # Remove leftover control markers like <|end|> or <|assistant|>
    text = re.sub(r"<\|.*?\|>", "", text)

    # Clean extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove redundant prefixes
    if text.lower().startswith("assistant:"):
        text = text.split(":", 1)[1].strip()

    # Keep only the first concise sentence if it’s too long
    if len(text.split(".")) > 3:
        text = ".".join(text.split(".")[:2]).strip() + "."

    return text


# -----------------------
# Pydantic request models
# -----------------------
class KnowledgeIn(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}

class ChatIn(BaseModel):
    query: str
    conversation_mode: str = "casual"
    relevance_threshold: float = 0.2

class FeedbackIn(BaseModel):
    query: str
    response: str
    score: float


# -----------------------
# Endpoints
# -----------------------
@app.post("/persona/{persona_id}/init")
def init_persona(persona_id: int):
    if persona_id not in persona_systems:
        persona_systems[persona_id] = PersonaRAGSystem(persona_id)
        logger.info(f"Initialized PersonaRAGSystem for persona {persona_id}")
    return {"ok": True, "persona_id": persona_id}


@app.post("/persona/{persona_id}/knowledge/add")
def add_knowledge(persona_id: int, body: KnowledgeIn):
    if persona_id not in persona_systems:
        raise HTTPException(status_code=404, detail="Persona not initialized")
    doc_id = persona_systems[persona_id].add_knowledge(body.text, body.metadata)
    return {"ok": True, "doc_id": doc_id}


@app.post("/persona/{persona_id}/chat")
def chat(persona_id: int, body: ChatIn):
    if persona_id not in persona_systems:
        raise HTTPException(status_code=404, detail="Persona not initialized")

    result = persona_systems[persona_id].generate_response(
        body.query,
        conversation_mode=body.conversation_mode,
        relevance_threshold=body.relevance_threshold
    )

    # Clean AI response text before returning
    if "response" in result:
        result["response"] = clean_response_text(result["response"])

    return result

# -----------------------
# Feedback endpoint (Reward-based only)
# -----------------------
feedback_counts: Dict[int, Dict[str, int]] = {}  # Store reward counts per persona


@app.post("/persona/{persona_id}/feedback")
def feedback(persona_id: int, body: FeedbackIn):
    """
    Reward-based feedback system.
    Simply counts the number of positive and negative rewards for each persona.
    """
    if persona_id not in persona_systems:
        raise HTTPException(status_code=404, detail="Persona not initialized")

    if persona_id not in feedback_counts:
        feedback_counts[persona_id] = {"likes": 0, "dislikes": 0}

    # Reward logic: +1 for positive, +1 dislike for negative
    if body.score > 0:
        feedback_counts[persona_id]["likes"] += 1
    else:
        feedback_counts[persona_id]["dislikes"] += 1

    logger.info(
        f"Feedback for Persona {persona_id}: 👍={feedback_counts[persona_id]['likes']} 👎={feedback_counts[persona_id]['dislikes']}"
    )

    return {
        "ok": True,
        "persona_id": persona_id,
        "feedback_summary": feedback_counts[persona_id],
        "message": "Feedback recorded (reward-based only, no learning applied)."
    }
@app.get("/persona/{persona_id}/feedback/stats")
def get_feedback_stats(persona_id: int):
    """
    Retrieve total likes/dislikes for this persona.
    """
    if persona_id not in feedback_counts:
        return {"persona_id": persona_id, "likes": 0, "dislikes": 0}
    return {"persona_id": persona_id, **feedback_counts[persona_id]}

@app.post("/persona/{persona_id}/feedback")
def feedback(persona_id: int, body: FeedbackIn):
    if persona_id not in persona_systems:
        raise HTTPException(status_code=404, detail="Persona not initialized")
    feedback_data = persona_systems[persona_id].update_from_feedback(
        body.query, body.response, body.score
    )
    return {"ok": True, "feedback": feedback_data}


@app.get("/persona/{persona_id}/knowledge_gaps")
def knowledge_gaps(persona_id: int):
    if persona_id not in persona_systems:
        raise HTTPException(status_code=404, detail="Persona not initialized")
    gaps = persona_systems[persona_id].get_knowledge_gaps()
    return {"gaps": gaps}

@app.get("/persona/{persona_id}/knowledge")
def get_all_knowledge(persona_id: int):
    """
    Get all stored knowledge items for the given persona.
    """
    if persona_id not in persona_systems:
        raise HTTPException(status_code=404, detail="Persona not initialized")
    
    knowledge_list = persona_systems[persona_id].get_all_knowledge()
    return {"persona_id": persona_id, "knowledge": knowledge_list}

@app.get("/persona/{persona_id}/export")
def export_knowledge(persona_id: int):
    if persona_id not in persona_systems:
        raise HTTPException(status_code=404, detail="Persona not initialized")
    data = persona_systems[persona_id].export_knowledge()
    return data


@app.post("/persona/{persona_id}/import")
def import_knowledge(persona_id: int, knowledge_data: Dict[str, Any]):
    if persona_id not in persona_systems:
        persona_systems[persona_id] = PersonaRAGSystem(persona_id)
    persona_systems[persona_id].import_knowledge(knowledge_data)
    return {"ok": True}

@app.delete("/persona/{persona_id}/knowledge/{index}")
def delete_knowledge(persona_id: int, index: int):
    """
    Delete a specific knowledge entry by index.
    """
    if persona_id not in persona_systems:
        raise HTTPException(status_code=404, detail="Persona not initialized")

    success = persona_systems[persona_id].delete_knowledge(index)
    if not success:
        raise HTTPException(status_code=404, detail=f"Knowledge index {index} not found")

    return {"ok": True, "deleted_index": index}

@app.get("/health")
def health():
    return {"status": "ok"}
