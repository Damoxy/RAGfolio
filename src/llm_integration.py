"""
LLM Integration using OpenRouter API for RAG responses
"""
import os
import json
from typing import List, Dict, Any, Optional
import logging
import requests
from datetime import datetime

try:
    from .config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL
except ImportError:
    # For direct execution
    from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMIntegration:
    """Handle LLM interactions for RAG responses"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_BASE_URL
        
        if not self.api_key:
            logger.warning("No OpenRouter API key provided. LLM features will be disabled.")
    
    def generate_response(self, query: str, search_results: List[Dict[str, Any]], 
                        response_type: str = "answer") -> Dict[str, Any]:
        """
        Generate a response using retrieved context
        
        Args:
            query: User's question/query
            search_results: List of search results from retriever
            response_type: Type of response ("answer", "summary", "analysis")
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.api_key:
            return {
                "response": "LLM integration not available (no API key configured)",
                "error": "missing_api_key"
            }
        
        # Build context from search results
        context = self._build_context(search_results)
        
        # Create prompt based on response type
        prompt = self._create_prompt(query, context, response_type)
        
        # Generate response
        try:
            response = self._call_openrouter(prompt)
            
            # Build response metadata
            sources = self._extract_sources(search_results)
            
            return {
                "response": response,
                "sources": sources,
                "context_used": len(search_results),
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response_type": response_type
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return {
                "response": f"Error generating response: {str(e)}",
                "error": str(e)
            }
    
    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Build context string from search results"""
        context_parts = []
        
        for i, result in enumerate(search_results[:10]):  # Limit context size
            source_info = f"Source {i+1}: {result['title']} ({result['file_name']})"
            content = result['content'].strip()
            
            context_parts.append(f"{source_info}\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, response_type: str) -> str:
        """Create prompt based on response type"""
        
        base_instructions = """You are a helpful AI assistant that answers questions based on provided context from a personal knowledge base. The context comes from documents including notes, transcripts, and writings."""
        
        if response_type == "answer":
            prompt = f"""{base_instructions}

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, say so clearly. Include relevant quotes or references when appropriate. Be concise but thorough.

Answer:"""

        elif response_type == "summary":
            prompt = f"""{base_instructions}

Context:
{context}

Query: {query}

Please provide a concise summary of the relevant information from the context that relates to the query. Focus on the key points and main ideas.

Summary:"""

        elif response_type == "analysis":
            prompt = f"""{base_instructions}

Context:
{context}

Topic for Analysis: {query}

Please analyze the provided context in relation to the topic. Look for patterns, themes, connections, and insights. Provide a thoughtful analysis that synthesizes the information.

Analysis:"""

        else:
            prompt = f"""{base_instructions}

Context:
{context}

Request: {query}

Please respond appropriately to the request based on the provided context.

Response:"""
        
        return prompt
    
    def _call_openrouter(self, prompt: str) -> str:
        """Make API call to OpenRouter"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "/RAGfolio",  # Optional
            "X-Title": "RAGfolio"  # Optional
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"Unexpected response format: {result}")
    
    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information for citations"""
        sources = []
        
        for i, result in enumerate(search_results):
            sources.append({
                "index": i + 1,
                "title": result['title'],
                "file_name": result['file_name'],
                "file_path": result['file_path'],
                "chunk_index": result['chunk_index'],
                "similarity_score": result['similarity_score'],
                "final_score": result['final_score'],
                "topics": result['topics'],
                "priority": result['priority'],
                "modified_time": result['modified_time']
            })
        
        return sources
    
    def generate_summary(self, document_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of a document from its chunks"""
        if not document_chunks:
            return {"response": "No content to summarize", "error": "empty_content"}
        
        # Combine chunks into context
        content = "\n\n".join([chunk['content'] for chunk in document_chunks])
        document_title = document_chunks[0]['title']
        
        prompt = f"""Please provide a comprehensive summary of the following document:

Title: {document_title}

Content:
{content}

Create a well-structured summary that captures the main themes, key points, and important insights. Organize the summary with clear sections if appropriate.

Summary:"""
        
        try:
            response = self._call_openrouter(prompt)
            
            return {
                "response": response,
                "document_title": document_title,
                "chunk_count": len(document_chunks),
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                "response": f"Error generating summary: {str(e)}",
                "error": str(e)
            }
    
    def suggest_questions(self, search_results: List[Dict[str, Any]]) -> List[str]:
        """Generate suggested questions based on search results"""
        if not self.api_key or not search_results:
            return []
        
        # Build brief context
        topics = set()
        titles = []
        
        for result in search_results[:5]:
            topics.update(result['topics'])
            titles.append(result['title'])
        
        context_summary = f"Topics: {', '.join(list(topics)[:10])}\nDocuments: {', '.join(titles[:5])}"
        
        prompt = f"""Based on the following context from a personal knowledge base, suggest 5 interesting and relevant questions that could be asked:

{context_summary}

Generate questions that would help explore the content further, make connections between ideas, or dive deeper into specific topics. Format as a simple numbered list.

Questions:"""
        
        try:
            response = self._call_openrouter(prompt)
            
            # Parse questions from response
            lines = response.strip().split('\n')
            questions = []
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Clean up question formatting
                    question = line.split('.', 1)[-1].strip()
                    question = question.lstrip('-•').strip()
                    if question and '?' in question:
                        questions.append(question)
            
            return questions[:5]
        
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter"""
        if not self.api_key:
            return []
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            models = response.json()
            
            # Filter for chat models and add relevant info
            chat_models = []
            for model in models.get('data', []):
                if 'chat' in model.get('id', '').lower() or 'gpt' in model.get('id', '').lower():
                    chat_models.append({
                        'id': model.get('id'),
                        'name': model.get('name', model.get('id')),
                        'description': model.get('description', ''),
                        'pricing': model.get('pricing', {})
                    })
            
            return chat_models
        
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []
