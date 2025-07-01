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
# Add OpenAI import
try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates question-answer pairs using LLM models (transformers or OpenAI)."""
    
    def __init__(self, num_questions_per_chunk: int = 3, 
                 llm_model: str = "Qwen/Qwen2.5-32B-Instruct",
                 max_length: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 offload_model: bool = True,
                 llm_backend: str = "transformers",
                 openai_api_key: Optional[str] = None,
                 openai_api_base: Optional[str] = None):
        self.num_questions_per_chunk = num_questions_per_chunk
        self.llm_model = llm_model
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.offload_model = offload_model
        self.llm_backend = llm_backend
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        
        logger.info(f"QAGenerator initialized with:")
        logger.info(f"  - LLM Model: {llm_model}")
        logger.info(f"  - Backend: {llm_backend}")
        logger.info(f"  - Questions per chunk: {num_questions_per_chunk}")
        logger.info(f"  - Temperature: {temperature}")
        logger.info(f"  - Top-p: {top_p}")
        logger.info(f"  - Max length: {max_length}")
        logger.info(f"  - Do sample: {do_sample}")
        logger.info(f"  - Offload model: {offload_model}")
        if llm_backend == "openai":
            logger.info(f"  - OpenAI API Key: {'set' if openai_api_key else 'not set'}")
            logger.info(f"  - OpenAI API Base: {openai_api_base}")
        
        self.pipeline = None
        self.device = self._get_device() if llm_backend == "transformers" else None
        self.model_loaded = False
        
        if not self.offload_model and self.llm_backend == "transformers":
            self._load_llm()
        if self.llm_backend == "openai" and openai is None:
            logger.error("OpenAI SDK is not installed. Please install openai package.")
            raise ImportError("openai package is required for OpenAI backend.")
        if self.llm_backend == "openai" and self.openai_api_key:
            # Initialize OpenAI client with new API format
            self.openai_client = openai.OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_api_base
            )
        else:
            self.openai_client = None
    
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
        """Load the LLM model and create pipeline."""
        if self.model_loaded or self.llm_backend != "transformers":
            return
            
        logger.info(f"Loading LLM model: {self.llm_model} (backend: transformers)")
        try:
            logger.info("Creating text generation pipeline...")
            self.pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                model_kwargs={"torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float16},
                device=self.device
            )
            logger.info("Pipeline created successfully")
            
            self.model_loaded = True
            logger.info(f"LLM model loaded successfully: {self.llm_model}")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            logger.warning("LLM model not available for QA generation")
            self.pipeline = None
            self.model_loaded = False
    
    def _unload_llm(self):
        """Unload the LLM model to free memory."""
        if not self.model_loaded or not self.offload_model or self.llm_backend != "transformers":
            return
            
        logger.info("Unloading LLM model to free memory...")
        
        if self.pipeline is not None:
            # Delete pipeline
            del self.pipeline
            self.pipeline = None
            
        self.model_loaded = False
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
            
        logger.info("LLM model unloaded successfully")
    
    def generate_qa_pairs(self, text_chunks: List[str], save_callback=None) -> List[Dict[str, Any]]:
        """Generate question-answer pairs from text chunks."""
        logger.info(f"Starting QA generation for {len(text_chunks)} text chunks")
        
        # Load model if needed
        if self.offload_model and not self.model_loaded and self.llm_backend == "transformers":
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
                
                # Save files incrementally if callback is provided
                if save_callback:
                    save_callback(chunk, chunk_qa_pairs, i)
                
                # Log progress every 10 chunks
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(text_chunks)} chunks processed, {total_questions_generated} QA pairs generated so far")
                    
                    # Unload model periodically to free memory if offloading is enabled
                    if self.offload_model and (i + 1) % 20 == 0 and self.llm_backend == "transformers":
                        self._unload_llm()
                        logger.info("Model unloaded to free memory. Will reload when needed.")
            
            logger.info(f"QA generation completed. Total QA pairs generated: {len(qa_pairs)}")
            logger.info(f"Average QA pairs per chunk: {len(qa_pairs) / len(text_chunks):.1f}")
            
        finally:
            # Always unload model after generation if offloading is enabled
            if self.offload_model and self.llm_backend == "transformers":
                self._unload_llm()
        
        return qa_pairs
    
    def _generate_qa_for_chunk(self, chunk: str, real_index: int) -> List[Dict[str, Any]]:
        """Generate QA pairs for a single text chunk, using the real dataset index for IDs and filenames."""
        logger.debug(f"Generating QA pairs for real dataset index {real_index}")
        
        # Ensure model is loaded
        if self.offload_model and not self.model_loaded and self.llm_backend == "transformers":
            self._load_llm()
        qa_pairs = []
        logger.debug("Generating context-aware questions...")
        for q_idx in range(self.num_questions_per_chunk):
            logger.debug(f"Generating question {q_idx + 1}/{self.num_questions_per_chunk}")
            if self.llm_backend == "transformers" and self.pipeline is not None and self.model_loaded:
                question = self._generate_question_from_context_transformers(chunk, q_idx)
                answer = self._generate_answer_from_question_context_transformers(chunk, question)
            elif self.llm_backend == "openai":
                question = self._generate_question_from_context_openai(chunk, q_idx)
                answer = self._generate_answer_from_question_context_openai(chunk, question)
            else:
                logger.warning(f"LLM model not available for real index {real_index}, skipping QA generation")
                continue
            qa_pair = {
                "id": f"dataset_{real_index}_question_{q_idx}",
                "question": question,
                "answer": answer,
                "context": chunk,
                "question_index": q_idx,
                "chunk_index": real_index
            }
            qa_pairs.append(qa_pair)
            logger.debug(f"Generated QA pair {q_idx + 1}: {len(question)} chars question, {len(answer)} chars answer")
        logger.debug(f"Generated {len(qa_pairs)} QA pairs for real dataset index {real_index}")
        return qa_pairs
    
    def _generate_question_from_context_transformers(self, context: str, question_index: int) -> str:
        """Generate a self-contained question based on the context that doesn't require the original context to answer."""
        try:
            logger.debug(f"Generating self-contained question from context {question_index + 1}")
            
            # Create instruction for self-contained question generation
            instruction = """Based on the following Arabic text, generate one self-contained question in Arabic that includes all necessary information to answer it without needing the original context. 

The question should:
1. Be completely self-contained - include all relevant details, names, dates, facts, or context needed to answer it
2. Be natural and conversational in Arabic
3. Cover various types: factual, analytical, comparative, causal, hypothetical, opinion-based, evaluative, etc.
4. Be specific enough that someone who hasn't read the original text could still answer it
5. Include any necessary background information or context within the question itself

Examples of good self-contained questions:
- "ما هي العاصمة السياسية والاقتصادية لمصر؟" (What is the political and economic capital of Egypt?)
- "كيف يؤثر تغير المناخ على الزراعة في منطقة الشرق الأوسط؟" (How does climate change affect agriculture in the Middle East?)
- "ما هي الفوائد الصحية للتمر على جسم الإنسان؟" (What are the health benefits of dates on the human body?)

The response should be only the question in Arabic — no explanations, translations, or additional commentary."""
            
            # Create messages for the conversation
            messages = [
                {"role": "user", "content": f"{instruction}\n\nText: {context}"}
            ]

            logger.debug(f"Generating self-contained question from context...")
            
            # Generate response using pipeline
            outputs = self.pipeline(
                messages,
                max_new_tokens=2048,
            )
            
            # Extract the generated question
            question = outputs[0]["generated_text"][-1]["content"].strip()
            
            # Clean up the question
            question = self._clean_question(question)
            
            logger.debug(f"LLM generated self-contained question: {question[:100]}...")
            
            return question
            
        except Exception as e:
            logger.error(f"Error generating self-contained question from context with LLM: {e}")
            return ""
    
    def _generate_answer_from_question_context_transformers(self, context: str, question: str) -> str:
        """Generate a comprehensive answer based on the question and context."""
        try:
            logger.debug(f"Generating comprehensive answer for self-contained question: {question[:50]}...")
            
            # Create instruction for comprehensive answer generation
            instruction = """Based on the following Arabic text, provide a comprehensive and detailed answer to the given question in Arabic. 

The answer should:
1. Be thorough and well-structured
2. Include relevant details, examples, and explanations
3. Be written in clear, natural Arabic
4. Provide a complete response that fully addresses the question
5. Be educational and informative
6. Use proper Arabic grammar and style

Focus on providing a complete answer that would satisfy someone asking this question, even if they don't have access to the original text."""
            
            # Create messages for the conversation
            messages = [
                {"role": "user", "content": f"{instruction}\n\nText: {context}\n\nQuestion: {question}"}
            ]
            
            # Generate response using pipeline
            outputs = self.pipeline(
                messages,
                max_new_tokens=200,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample
            )
            
            # Extract the generated answer
            answer = outputs[0]["generated_text"][-1]["content"].strip()
            
            # Clean up the answer
            answer = self._clean_answer(answer)
            
            logger.debug(f"LLM generated comprehensive answer: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating comprehensive answer with LLM: {e}")
            return ""
    
    def _generate_question_from_context_openai(self, context: str, question_index: int) -> str:
        try:
            prompt = f"Generate a self-contained question in Arabic based on the following text. Include all relevant details.\n\nText: {context}"
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=self.temperature,
                top_p=self.top_p,
                n=1
            )
            question = response.choices[0].message.content.strip()
            question = self._clean_question(question)
            logger.debug(f"OpenAI generated self-contained question: {question[:100]}...")
            return question
        except Exception as e:
            logger.error(f"Error generating question from context with OpenAI: {e}")
            return ""
    
    def _generate_answer_from_question_context_openai(self, context: str, question: str) -> str:
        try:
            instruction = """Based on the following Arabic text, provide a comprehensive and detailed answer to the given question in Arabic. \
\nThe answer should:\n1. Be thorough and well-structured\n2. Include relevant details, examples, and explanations\n3. Be written in clear, natural Arabic\n4. Provide a complete response that fully addresses the question\n5. Be educational and informative\n6. Use proper Arabic grammar and style\n\nFocus on providing a complete answer that would satisfy someone asking this question, even if they don't have access to the original text."""
            prompt = f"{instruction}\n\nText: {context}\n\nQuestion: {question}"
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=self.temperature,
                top_p=self.top_p,
                n=1
            )
            answer = response.choices[0].message.content.strip()
            answer = self._clean_answer(answer)
            logger.debug(f"OpenAI generated comprehensive answer: {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer with OpenAI: {e}")
            return ""
    
    def _clean_question(self, question: str) -> str:
        # """Clean up generated question."""
        # # Remove extra whitespace and newlines
        # question = re.sub(r'\s+', ' ', question.strip())
        
        # # Ensure it ends with a question mark
        # if not question.endswith('?'):
        #     question += '?'
        
        # # Remove any incomplete sentences
        # if len(question.split()) < 3:
        #     return ""
        
        return question
    
    def _clean_answer(self, answer: str) -> str:
        """Clean up generated answer."""
        # Remove extra whitespace and newlines
        # answer = re.sub(r'\s+', ' ', answer.strip())
        
        # # Remove any incomplete sentences at the end
        # sentences = answer.split('.')
        # if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
        #     answer = '.'.join(sentences[:-1]) + '.'
        
        # Truncate if too long
        # if len(answer) > 500:
        #     answer = answer[:500] + "..."
        
        return answer
    
    def generate_instruction_format(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert QA pairs to instruction format for training."""
        logger.info("Converting QA pairs to instruction format...")
        instructions = []
        
        for qa in qa_pairs:
            instruction = {
                "instruction": qa["question"],
                "input": "",  # Empty since questions are now self-contained
                "output": qa["answer"],
                "id": qa["id"]
            }
            instructions.append(instruction)
        
        logger.info(f"Converted {len(instructions)} QA pairs to instruction format")
        return instructions
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if self.offload_model:
            self._unload_llm() 