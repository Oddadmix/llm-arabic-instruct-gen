"""
Question-Answer generation module using transformers library for LLM-based generation.
"""

import random
import re
from typing import List, Dict, Any, Optional
import logging
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates question-answer pairs using LLM models."""
    
    def __init__(self, num_questions_per_chunk: int = 3, 
                 question_types: Optional[List[str]] = None,
                 llm_model: str = "Qwen/Qwen2.5-32B-Instruct",
                 max_length: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 offload_model: bool = True):
        self.num_questions_per_chunk = num_questions_per_chunk
        self.question_types = question_types or ["what", "how", "why"]
        self.llm_model = llm_model
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.offload_model = offload_model
        
        logger.info(f"QAGenerator initialized with:")
        logger.info(f"  - LLM Model: {llm_model}")
        logger.info(f"  - Questions per chunk: {num_questions_per_chunk}")
        logger.info(f"  - Question types: {question_types}")
        logger.info(f"  - Temperature: {temperature}")
        logger.info(f"  - Top-p: {top_p}")
        logger.info(f"  - Max length: {max_length}")
        logger.info(f"  - Do sample: {do_sample}")
        logger.info(f"  - Offload model: {offload_model}")
        
        # Initialize LLM components
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        self.model_loaded = False
        
        # Question templates for different types
        self.question_templates = {
            "what": [
                "What is {topic}?",
                "What does {topic} refer to?",
                "What are the main aspects of {topic}?",
                "What is the definition of {topic}?"
            ],
            "how": [
                "How does {topic} work?",
                "How is {topic} implemented?",
                "How can {topic} be achieved?",
                "How does {topic} function?"
            ],
            "why": [
                "Why is {topic} important?",
                "Why does {topic} matter?",
                "Why should we consider {topic}?",
                "Why is {topic} relevant?"
            ]
        }
        
        # Only load model if not offloading
        if not self.offload_model:
            self._load_llm()
    
    def _get_device(self) -> torch.device:
        """Determine the device to use for model inference."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        return device
    
    def _load_llm(self):
        """Load the LLM model and tokenizer."""
        if self.model_loaded:
            logger.debug("Model already loaded")
            return
            
        logger.info(f"Loading LLM model: {self.llm_model}")
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            logger.info("Tokenizer loaded successfully")
            
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(self.llm_model)
            logger.info("Model loaded successfully")
            
            logger.info(f"Moving model to device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model moved to device and set to evaluation mode")
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            self.model_loaded = True
            logger.info(f"LLM model loaded successfully: {self.llm_model}")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            logger.warning("Falling back to template-based question generation")
            self.tokenizer = None
            self.model = None
            self.model_loaded = False
    
    def _unload_llm(self):
        """Unload the LLM model to free memory."""
        if not self.model_loaded or not self.offload_model:
            return
            
        logger.info("Unloading LLM model to free memory...")
        
        if self.model is not None:
            # Move model to CPU first to free GPU memory
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            
            # Delete model
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        self.model_loaded = False
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
            
        logger.info("LLM model unloaded successfully")
    
    def generate_qa_pairs(self, text_chunks: List[str]) -> List[Dict[str, Any]]:
        """Generate question-answer pairs from text chunks."""
        logger.info(f"Starting QA generation for {len(text_chunks)} text chunks")
        
        # Load model if needed
        if self.offload_model and not self.model_loaded:
            self._load_llm()
        
        qa_pairs = []
        total_questions_generated = 0
        
        try:
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing chunk {i + 1}/{len(text_chunks)} (length: {len(chunk)} characters)")
                
                chunk_qa_pairs = self._generate_qa_for_chunk(chunk, i)
                qa_pairs.extend(chunk_qa_pairs)
                total_questions_generated += len(chunk_qa_pairs)
                
                logger.info(f"Chunk {i + 1}: Generated {len(chunk_qa_pairs)} QA pairs")
                
                # Log progress every 10 chunks
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(text_chunks)} chunks processed, {total_questions_generated} QA pairs generated so far")
                    
                    # Unload model periodically to free memory if offloading is enabled
                    if self.offload_model and (i + 1) % 20 == 0:
                        self._unload_llm()
                        logger.info("Model unloaded to free memory. Will reload when needed.")
            
            logger.info(f"QA generation completed. Total QA pairs generated: {len(qa_pairs)}")
            logger.info(f"Average QA pairs per chunk: {len(qa_pairs) / len(text_chunks):.1f}")
            
        finally:
            # Always unload model after generation if offloading is enabled
            if self.offload_model:
                self._unload_llm()
        
        return qa_pairs
    
    def _generate_qa_for_chunk(self, chunk: str, chunk_index: int) -> List[Dict[str, Any]]:
        """Generate QA pairs for a single text chunk."""
        logger.debug(f"Generating QA pairs for chunk {chunk_index}")
        
        # Ensure model is loaded
        if self.offload_model and not self.model_loaded:
            self._load_llm()
        
        qa_pairs = []
        
        # Extract key topics from the chunk
        logger.debug("Extracting topics from chunk...")
        topics = self._extract_topics(chunk)
        logger.debug(f"Found {len(topics)} potential topics: {topics[:5]}...")  # Log first 5 topics
        
        # Generate questions for each topic
        questions_to_generate = min(len(topics), self.num_questions_per_chunk)
        logger.debug(f"Will generate {questions_to_generate} questions from {len(topics)} topics")
        
        for topic_idx, topic in enumerate(topics[:self.num_questions_per_chunk]):
            question_type = random.choice(self.question_types)
            logger.debug(f"Generating {question_type} question for topic: {topic}")
            
            if self.model is not None and self.model_loaded:
                # Use LLM for question generation
                logger.debug("Using LLM for question generation...")
                question = self._generate_question_with_llm(chunk, topic, question_type)
                answer = self._generate_answer_with_llm(chunk, topic, question)
            else:
                # Fallback to template-based generation
                logger.debug("Using template-based generation...")
                question = self._generate_question(topic, question_type)
                answer = self._generate_answer(chunk, topic, question)
            
            qa_pair = {
                "id": f"chunk_{chunk_index}_topic_{topic_idx}",
                "question": question,
                "answer": answer,
                "context": chunk,
                "question_type": question_type,
                "topic": topic,
                "chunk_index": chunk_index
            }
            
            qa_pairs.append(qa_pair)
            logger.debug(f"Generated QA pair {topic_idx + 1}: {len(question)} chars question, {len(answer)} chars answer")
        
        logger.debug(f"Generated {len(qa_pairs)} QA pairs for chunk {chunk_index}")
        return qa_pairs
    
    def _generate_question_with_llm(self, context: str, topic: str, question_type: str) -> str:
        """Generate a question using the LLM model."""
        try:
            logger.debug(f"Generating {question_type} question for topic '{topic}' using LLM")
            
            # Create prompt for question generation
            prompt = f"""Based on the following context, generate a {question_type} question about {topic}:

Context: {context[:500]}...

Question:"""
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = inputs.to(self.device)
            logger.debug(f"Tokenized input: {inputs.shape[1]} tokens")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated question
            question = response[len(prompt):].strip()
            
            # Clean up the question
            question = self._clean_question(question)
            
            logger.debug(f"LLM generated question: {question[:100]}...")
            
            return question if question else self._generate_question(topic, question_type)
            
        except Exception as e:
            logger.error(f"Error generating question with LLM: {e}")
            return self._generate_question(topic, question_type)
    
    def _generate_answer_with_llm(self, context: str, topic: str, question: str) -> str:
        """Generate an answer using the LLM model."""
        try:
            logger.debug(f"Generating answer for question: {question[:50]}...")
            
            # Create prompt for answer generation
            prompt = f"""Based on the following context, answer this question:

Context: {context[:800]}...

Question: {question}

Answer:"""
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = inputs.to(self.device)
            logger.debug(f"Tokenized input: {inputs.shape[1]} tokens")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated answer
            answer = response[len(prompt):].strip()
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            
            logger.debug(f"LLM generated answer: {answer[:100]}...")
            
            return answer if answer else self._generate_answer(context, topic, question)
            
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            return self._generate_answer(context, topic, question)
    
    def _clean_question(self, question: str) -> str:
        """Clean up generated question."""
        # Remove extra whitespace and newlines
        question = re.sub(r'\s+', ' ', question.strip())
        
        # Ensure it ends with a question mark
        if not question.endswith('?'):
            question += '?'
        
        # Remove any incomplete sentences
        if len(question.split()) < 3:
            return ""
        
        return question
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up generated answer."""
        # Remove extra whitespace and newlines
        answer = re.sub(r'\s+', ' ', answer.strip())
        
        # Remove any incomplete sentences at the end
        sentences = answer.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            answer = '.'.join(sentences[:-1]) + '.'
        
        # Truncate if too long
        if len(answer) > 500:
            answer = answer[:500] + "..."
        
        return answer
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from text for question generation."""
        logger.debug("Extracting topics from text...")
        
        # Simple topic extraction - can be improved with NLP
        words = text.split()
        
        # Filter for potential topics (nouns, longer words)
        topics = []
        for word in words:
            word = word.strip('.,!?;:()[]{}').lower()
            if len(word) > 4 and word.isalpha():
                topics.append(word)
        
        # Remove duplicates and limit
        unique_topics = list(set(topics))[:10]
        logger.debug(f"Extracted {len(unique_topics)} unique topics")
        
        return unique_topics
    
    def _generate_question(self, topic: str, question_type: str) -> str:
        """Generate a question based on topic and type (fallback method)."""
        templates = self.question_templates.get(question_type, self.question_templates["what"])
        template = random.choice(templates)
        return template.format(topic=topic)
    
    def _generate_answer(self, context: str, topic: str, question: str) -> str:
        """Generate an answer based on context and question (fallback method)."""
        # Simple answer generation - can be improved with LLM integration
        sentences = context.split('.')
        
        # Find sentences that mention the topic
        relevant_sentences = []
        for sentence in sentences:
            if topic.lower() in sentence.lower():
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Use the first relevant sentence as answer
            answer = relevant_sentences[0]
            if len(answer) > 200:  # Truncate if too long
                answer = answer[:200] + "..."
            return answer
        else:
            # Fallback: use a portion of the context
            return context[:200] + "..." if len(context) > 200 else context
    
    def generate_instruction_format(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert QA pairs to instruction format for training."""
        logger.info("Converting QA pairs to instruction format...")
        instructions = []
        
        for qa in qa_pairs:
            instruction = {
                "instruction": qa["question"],
                "input": qa["context"],
                "output": qa["answer"],
                "id": qa["id"]
            }
            instructions.append(instruction)
        
        logger.info(f"Converted {len(instructions)} QA pairs to instruction format")
        return instructions
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'offload_model') and self.offload_model:
            self._unload_llm() 