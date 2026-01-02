"""LangChain chains for question processing: correction, rephrasing, and splitting."""
import sys
from pathlib import Path

# Ensure project root is in path for local imports
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
try:
    from langchain_ollama import ChatOllama
except ImportError:
    # Fallback for older versions
    from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from config.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TEMPERATURE_CORRECTION,
)
from utils.performance_profiler import get_profiler


class QuestionProcessingResult(BaseModel):
    """Structured output for question processing."""
    corrected: str = Field(description="Spelling-corrected question")
    rephrased: str = Field(description="Rephrased and improved question")
    parts: List[str] = Field(description="List of decomposed question parts")


class QuestionCorrectionChain:
    """Chain for correcting spelling mistakes in questions."""
    
    def __init__(self, llm: Optional[ChatOllama] = None):
        """
        Initialize correction chain.
        
        Args:
            llm: Optional pre-initialized LLM instance (for reuse)
        """
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE_CORRECTION,
            )
        
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template="""Correct ALL spelling mistakes in the following question. 
Return ONLY the corrected question, nothing else.

Question: {question}

Corrected Question:"""
        )
        
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser
    
    def correct(self, question: str) -> str:
        """Correct spelling in a question."""
        try:
            result = self.chain.invoke({"question": question})
            return result.strip() if result else question.strip()
        except Exception:
            return question.strip()


class QuestionRephrasingChain:
    """Chain for rephrasing questions clearly and professionally."""
    
    def __init__(self, llm: Optional[ChatOllama] = None):
        """
        Initialize rephrasing chain.
        
        Args:
            llm: Optional pre-initialized LLM instance (for reuse)
        """
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE_CORRECTION,
            )
        
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a question refinement assistant for an educational search system.

Your task is to improve the clarity of the user's question WITHOUT changing
its original intent, scope, or difficulty level.

RULES:
- Do NOT add new requirements or sub-questions.
- Do NOT make the question more advanced or technical.
- Do NOT turn a simple question into an in-depth academic explanation.
- Preserve the original meaning exactly.
- Only fix spelling, grammar, and minor ambiguity.

If the question is already clear, return it unchanged.

OUTPUT:
Return ONLY the rephrased question. No explanations.

Original Question:
{question}

Rephrased Question:"""
        )
        
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser
    
    def rephrase(self, question: str) -> str:
        """Rephrase a question."""
        try:
            result = self.chain.invoke({"question": question})
            return result.strip() if result else question.strip()
        except Exception:
            return question.strip()


class QuestionSplittingChain:
    """Chain for splitting complex questions into sub-questions with semantic deduplication."""
    
    def __init__(self, llm: Optional[ChatOllama] = None):
        """
        Initialize splitting chain.
        
        Args:
            llm: Optional pre-initialized LLM instance (for reuse)
        """
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE_CORRECTION,
            )
        
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template="""Analyze the following question carefully. 

IMPORTANT RULES:
1. Only split if the question contains CLEARLY DISTINCT sub-intents (different topics, entities, or conditions).
2. Do NOT split if the question is a single coherent intent, even if it mentions multiple aspects of the same topic.
3. Only split on explicit conjunctions (e.g., "and", "compare", "difference between") when they introduce MEANINGFULLY DIFFERENT aspects.
4. Avoid creating semantically duplicate sub-questions that ask the same thing in different words.
5. If the question asks about one topic from multiple angles (e.g., "process and mechanism"), keep it as ONE question.

Examples:
- "What is photosynthesis and how does it work?" → Keep as ONE question (same topic, different aspects are part of the same intent)
- "What is photosynthesis and what is cellular respiration?" → Split into TWO questions (different topics)
- "Compare photosynthesis and cellular respiration" → Keep as ONE question (comparison is a single intent)
- "What is the difference between photosynthesis and cellular respiration?" → Keep as ONE question (difference is a single intent)

Question: {question}

Return ONLY a valid JSON array of strings. If the question should not be split, return a single-element array with the original question.

Output:"""
        )
        
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser
        self.use_semantic_check = False  # Disabled - no embedding model needed
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts (disabled - no embedding model)."""
        return 0.0  # Semantic checking disabled
    
    def _deduplicate_parts(self, parts: List[str], similarity_threshold: float = 0.90) -> List[str]:
        """
        Remove semantically duplicate parts from the list.
        
        Args:
            parts: List of question parts
            similarity_threshold: Minimum similarity score to consider as duplicate (0.0-1.0)
        
        Returns:
            Deduplicated list of parts
        """
        if len(parts) <= 1:
            return parts
        
        if not self.use_semantic_check:
            # If semantic checking is not available, do basic string-based deduplication
            unique_parts = []
            seen = set()
            for part in parts:
                part_lower = part.lower().strip()
                if part_lower not in seen:
                    seen.add(part_lower)
                    unique_parts.append(part)
            return unique_parts
        
        # Use semantic similarity to deduplicate
        unique_parts = []
        for i, part1 in enumerate(parts):
            is_duplicate = False
            
            # Check against all previously added unique parts
            for part2 in unique_parts:
                similarity = self._compute_semantic_similarity(part1, part2)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_parts.append(part1)
        
        return unique_parts
    
    def split(self, question: str) -> List[str]:
        """Split a question into parts, with semantic deduplication."""
        try:
            import json
            result = self.chain.invoke({"question": question})
            text = result.strip() if isinstance(result, str) else str(result).strip()
            
            # Try to parse as JSON array
            if text.startswith("```"):
                # Remove markdown code blocks
                lines = text.split("\n")
                text = "\n".join([l for l in lines if not l.startswith("```")])
            
            parts = json.loads(text)
            if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
                return [question]
            
            # Filter out empty parts
            parts = [p.strip() for p in parts if p.strip()]
            
            if not parts:
                return [question]
            
            # If only one part, return it
            if len(parts) == 1:
                return parts
            
            # Deduplicate semantically similar parts
            deduplicated = self._deduplicate_parts(parts, similarity_threshold=0.90)
            
            # If all parts were duplicates, return the first one
            if not deduplicated:
                return [parts[0]]
            
            # If deduplication resulted in a single part, return it
            if len(deduplicated) == 1:
                return deduplicated
            
            return deduplicated
            
        except Exception:
            # If parsing fails, return original question as single part
            return [question]


class CombinedQuestionProcessingChain:
    """Combined chain that performs correction, rephrasing, and splitting."""
    
    def __init__(self, llm: Optional[ChatOllama] = None):
        """
        Initialize combined question processing chain.
        
        Args:
            llm: Optional pre-initialized LLM instance (for reuse)
        """
        self.correction_chain = QuestionCorrectionChain(llm=llm)
        self.rephrasing_chain = QuestionRephrasingChain(llm=llm)
        self.splitting_chain = QuestionSplittingChain(llm=llm)
    
    def process(self, question: str) -> QuestionProcessingResult:
        """Process a question through all stages."""
        profiler = get_profiler()
        
        # Step 1: Correct spelling
        with profiler.stage("4a_text_correction", {"question_length": len(question)}):
            corrected = self.correction_chain.correct(question)
        
        # Step 2: Rephrase
        with profiler.stage("4b_text_rephrasing", {"corrected_length": len(corrected)}):
            rephrased = self.rephrasing_chain.rephrase(corrected)
        
        # Step 3: Split into parts
        with profiler.stage("4c_text_splitting", {"rephrased_length": len(rephrased)}):
            parts = self.splitting_chain.split(rephrased)
        
        return QuestionProcessingResult(
            corrected=corrected,
            rephrased=rephrased,
            parts=parts
        )

