from importlib import metadata
import os
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
import logging
from gpt4all import GPT4All
from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PersonaRAGSystem:
    def __init__(self, persona_id: int):
        self.persona_id = persona_id
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Initialize ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=f"{settings.VECTOR_DB_PATH}/persona_{persona_id}",
            settings=ChromaSettings(allow_reset=True)
        )

        # Create or get collection for this persona
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"persona_{persona_id}_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        # Local GPT4All setup
        model_path = r"C:\Users\pc\Desktop\pythonproject\models\nnnn.gguf"
        self.generator = GPT4All(model_path)

    # ----------------------------------------------------------------------
    # Knowledge Management
    # ----------------------------------------------------------------------

    def add_knowledge(self, text: str, metadata: Dict[str, Any]) -> str:
        try:
            embedding = self.embedding_model.encode(text).tolist()
            existing = self.collection.get()

            # Find the last inserted ID number
            if existing and 'ids' in existing and existing['ids']:
                last_id = existing['ids'][-1]              # e.g., "personal_15"
                try:
                    last_num = int(last_id.split('_')[-1])  # extract 15
                except ValueError:
                    last_num = 0
            else:
                last_num = 0

            # Generate the next sequential ID
            next_id = f"{metadata.get('category', 'general')}_{last_num + 1}"

            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[next_id]
            )

            logger.info(f"Added knowledge to persona {self.persona_id}: {next_id}")
            return next_id

        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            raise


    def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """Retrieve and clean all stored knowledge entries for this persona."""
        try:
            all_docs = self.collection.get(include=["documents", "metadatas"])
            knowledge_entries = []

            if not all_docs or not all_docs.get("documents"):
                return []

            ids = all_docs.get("ids", [f"item_{i}" for i in range(len(all_docs["documents"]))])

            for i, text in enumerate(all_docs["documents"]):
                if not text:
                    continue
                cleaned_text = text.replace("\n", " ").replace("\t", " ").strip()
                if cleaned_text.startswith("http") or len(cleaned_text.split()) < 3:
                    continue
                knowledge_entries.append({
                    "id": ids[i],
                    "text": cleaned_text,
                    "metadata": all_docs["metadatas"][i] if all_docs["metadatas"] else {}
                })

            return knowledge_entries

        except Exception as e:
            logger.error(f"Error retrieving all knowledge for persona {self.persona_id}: {e}")
            return []

    def delete_knowledge(self, index: int) -> bool:
        """Delete a specific knowledge entry by its index or ID."""
        try:
            all_docs = self.collection.get()
            ids = all_docs.get("ids", [])
            if not ids or index < 0 or index >= len(ids):
                return False

            target_id = ids[index]
            self.collection.delete(ids=[target_id])
            logger.info(f"Deleted knowledge from persona {self.persona_id}: {target_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting knowledge for persona {self.persona_id}: {e}")
            return False

    # ----------------------------------------------------------------------
    # Knowledge Export/Import
    # ----------------------------------------------------------------------

    def export_knowledge(self) -> Dict[str, Any]:
        all_data = self.collection.get()
        return {
            'persona_id': self.persona_id,
            'documents': all_data['documents'],
            'metadatas': all_data['metadatas'],
            'ids': all_data['ids']
        }

    def import_knowledge(self, knowledge_data: Dict[str, Any]):
        try:
            self.collection.delete()
            if knowledge_data['documents']:
                embeddings = [self.embedding_model.encode(doc).tolist() for doc in knowledge_data['documents']]
                self.collection.add(
                    embeddings=embeddings,
                    documents=knowledge_data['documents'],
                    metadatas=knowledge_data['metadatas'],
                    ids=knowledge_data['ids']
                )
            logger.info(f"Imported {len(knowledge_data['documents'])} knowledge items")
        except Exception as e:
            logger.error(f"Error importing knowledge: {e}")
            raise

    # ----------------------------------------------------------------------
    # Response Generation
    # ----------------------------------------------------------------------

    def retrieve_relevant_knowledge(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )

            knowledge_items = []
            for i, doc in enumerate(results['documents'][0]):
                knowledge_items.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i],
                    'id': f"item_{i}"
                })
            return knowledge_items
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return []

    def generate_response(self, query: str, conversation_mode: str = "casual", relevance_threshold: float = 0.2) -> Dict[str, Any]:
        try:
            relevant_knowledge = self.retrieve_relevant_knowledge(query)
            relevant_knowledge = [item for item in relevant_knowledge if item['similarity'] >= relevance_threshold]
            relevant_knowledge.sort(key=lambda x: x['similarity'], reverse=True)

            if not relevant_knowledge:
                return {
                    'response': "I don't have enough information to answer that question yet. Could you provide more details?",
                    'confidence': 0.1,
                    'sources': []
                }

            context = self._build_context(relevant_knowledge[:3], conversation_mode)
            response = self._generate_llm_response(query, context, conversation_mode)
            confidence = self._calculate_confidence(relevant_knowledge)

            return {
                'response': response,
                'confidence': confidence,
                'sources': [
                    {
                        'content': item['content'][:100] + "..." if len(item['content']) > 100 else item['content'],
                        'category': item['metadata'].get('category', 'unknown'),
                        'similarity': item['similarity']
                    }
                    for item in relevant_knowledge[:3]
                ]
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': "I'm sorry, I encountered an error while processing your question.",
                'confidence': 0.0,
                'sources': []
            }

    def _build_context(self, knowledge_items: List[Dict[str, Any]], mode: str) -> str:
        """
        Build a readable, memory-like context for the LLM instead of a rigid list.
        Converts stored knowledge into short descriptive sentences grouped by category.
        """
        if not knowledge_items:
            return "No specific background information is available for this persona yet."

        # Organize knowledge by category
        categorized = {}
        for item in knowledge_items:
            category = item['metadata'].get('category', 'general').lower()
            categorized.setdefault(category, []).append(item['content'])

        context_summary = "Here's what is known about this persona:\n"
        for category, entries in categorized.items():
            context_summary += f"\nIn terms of {category}:\n"
            for e in entries:
                cleaned = e.strip().replace("\n", " ").replace("  ", " ")
                if not cleaned.endswith('.'):
                    cleaned += '.'
                context_summary += f"- {cleaned}\n"

        # Adjust tone based on mode
        if mode == "casual":
            tone_hint = "Use a friendly, natural tone as if recalling what you remember about a friend."
        elif mode == "professional":
            tone_hint = "Maintain a formal and concise tone suitable for workplace or business discussions."
        else:
            tone_hint = "Be expressive and creative, but stay within the known facts."

        # Final instruction to keep model grounded
        context_summary += f"\n\nInstruction: {tone_hint} Only use the above facts to answer. If unsure, say you don’t know."

        return context_summary


    def _generate_llm_response(self, query: str, context: str, mode: str) -> str:
        """
        Generates a grounded response using GPT4All based only on the provided context.
        Avoids hallucination and keeps tone based on conversation mode.
        """

        mode_prompts = {
            "casual": "You are having a friendly, casual chat. Be natural and conversational.",
            "professional": "You are in a formal, professional discussion. Be clear, factual, and concise.",
            "creative": "Be expressive, use analogies and creative explanations while staying relevant."
        }

        # Instruction emphasizing grounded reasoning
        instruction = (
            "Use only the knowledge provided below to answer the question. "
            "Do NOT invent or assume any extra information. "
            "If the knowledge does not contain an answer, say so politely."
        )

        # Construct final grounded prompt
        prompt = f"""
{mode_prompts.get(mode, 'casual')}

{instruction}

Context (known facts):
{context}

Question:
{query}

Answer in a natural way that feels like a conversation:
"""

        try:
            response = self.generator.generate(
                prompt,
                max_tokens=200,
                temp=0.4,
                top_p=0.9
            ).strip()

            # Fallback handling
            if not response or response.lower().startswith("as an ai"):
                response = "I'm not entirely sure about that based on what I currently know."

            # Ensure no hallucinated additions
            if "based on the provided knowledge" in response.lower() or "from what i know" in response.lower():
                return response

            # Add subtle grounding reminder
            if not any(word in response.lower() for word in ["based on", "according to", "from what I know"]):
                response = f"According to what I know, {response}"

            return response

        except Exception as e:
            logger.error(f"Error generating local LLM response: {e}")
            return "I'm having trouble forming a grounded response right now. Try again later."


    def _calculate_confidence(self, knowledge_items: List[Dict[str, Any]]) -> float:
        if not knowledge_items:
            return 0.0
        max_similarity = max(item['similarity'] for item in knowledge_items)
        relevance_boost = min(len(knowledge_items) * 0.1, 0.3)
        return round(min(max_similarity + relevance_boost, 1.0), 2)

    # ----------------------------------------------------------------------
    # Feedback & Knowledge Gaps
    # ----------------------------------------------------------------------

    def update_from_feedback(self, query: str, response: str, feedback_score: float):
        logger.info(f"Received feedback for persona {self.persona_id}: {feedback_score}")
        return {
            'query': query,
            'response': response,
            'score': feedback_score,
            'persona_id': self.persona_id
        }

    def get_knowledge_gaps(self) -> List[str]:
        all_docs = self.collection.get()
        if not all_docs['metadatas']:
            return ["personal", "career", "interests", "values", "experiences"]

        category_counts = {}
        for metadata in all_docs['metadatas']:
            category = metadata.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1

        min_items = 3
        gaps = [category for category, count in category_counts.items() if count < min_items]

        standard_categories = ["personal", "career", "interests", "values", "experiences", "opinions"]
        for category in standard_categories:
            if category not in category_counts:
                gaps.append(category)

        return gaps


# ----------------------------------------------------------------------
# Optional Evaluator (for future RAG improvement)
# ----------------------------------------------------------------------

class RAGEvaluator:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def evaluate_response_quality(self, query: str, response: str, ground_truth: str = None) -> Dict[str, float]:
        scores = {}
        query_embedding = self.embedding_model.encode(query)
        response_embedding = self.embedding_model.encode(response)
        relevance = np.dot(query_embedding, response_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding)
        )
        scores['relevance'] = max(0, relevance)
        scores['length'] = self._evaluate_response_length(response)
        scores['coherence'] = self._evaluate_coherence(response)
        scores['overall'] = scores['relevance'] * 0.5 + scores['length'] * 0.2 + scores['coherence'] * 0.3
        return scores

    def _evaluate_response_length(self, response: str) -> float:
        word_count = len(response.split())
        if 10 <= word_count <= 100:
            return 1.0
        elif 5 <= word_count < 10 or 100 < word_count <= 200:
            return 0.7
        elif word_count < 5:
            return 0.3
        else:
            return 0.5

    def _evaluate_coherence(self, response: str) -> float:
        return 1.0  # Placeholder for semantic coherence scoring
