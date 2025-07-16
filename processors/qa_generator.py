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
        logger.debug("Generating all questions for chunk in one prompt...")
        
        # Generate all questions for this chunk in one prompt
        if self.llm_backend == "transformers" and self.pipeline is not None and self.model_loaded:
            questions = self._generate_all_questions_from_context_transformers(chunk)
        elif self.llm_backend == "openai":
            questions = self._generate_all_questions_from_context_openai(chunk)
        else:
            logger.warning(f"LLM model not available for real index {real_index}, skipping QA generation")
            return qa_pairs
        
        # Generate answers for each question individually
        for q_idx, question in enumerate(questions):
            logger.debug(f"Generating answer for question {q_idx + 1}/{len(questions)}")
            
            if self.llm_backend == "transformers" and self.pipeline is not None and self.model_loaded:
                answer = self._generate_answer_from_question_context_transformers(chunk, question)
            elif self.llm_backend == "openai":
                answer = self._generate_answer_from_question_context_openai(chunk, question)
            else:
                logger.warning(f"LLM model not available for answer generation, skipping")
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
    
    def _generate_all_questions_from_context_transformers(self, context: str) -> List[str]:
        """Generate all questions for a chunk in one prompt to ensure diversity."""
        try:
            logger.debug(f"Generating {self.num_questions_per_chunk} diverse questions from context in one prompt")
            
            # Create instruction for generating multiple diverse questions
            instruction = f"""Based on the following Arabic text, generate exactly {self.num_questions_per_chunk} diverse, self-contained questions in Arabic. CRITICAL REQUIREMENT: Each question must be answerable ONLY using the information explicitly provided in the given text chunk - no external knowledge should be required.
The questions should:

Be answerable exclusively from the provided text - every piece of information needed to answer the question must be present in the text chunk
Be completely self-contained - include all relevant details, names, dates, facts, or context mentioned in the text
Be natural and conversational in Arabic
Cover various cognitive levels and question types:

Factual/Recall: Direct information retrieval from the text (ما، من، متى، أين)
Comprehension: Understanding concepts explained in the text (كيف، لماذا)
Application: Using information presented in the text (ماذا لو، كيف يمكن)
Analysis: Analyzing relationships and connections within the text (ما العلاقة بين، قارن)
Synthesis: Combining ideas presented in the text (ما الاستنتاج، اربط بين)
Evaluation: Making judgments based on information in the text (بناءً على النص، قيّم)


Never require external knowledge - avoid questions about general facts, historical events, or information not mentioned in the text
Focus only on content, concepts, examples, data, and information explicitly stated in the provided text
Be diverse and non-repetitive - each question should focus on different aspects, details, or perspectives from the text
Vary in complexity, depth, and scope while staying within the text's boundaries
Use natural Arabic phrasing and question structures

Format your response as a numbered list:

[First question in Arabic]
[Second question in Arabic]
[Third question in Arabic]
... and so on for {self.num_questions_per_chunk} questions

IMPORTANT: All questions must reference specific information from the text. Examples of context-dependent questions:

Factual: "ما هي المكونات الثلاثة التي ذكرها الكاتب في الوصفة؟"
Analytical: "حسب النص، كيف تؤثر درجة الحرارة المذكورة على عملية النمو؟"
Comparative: "ما الفروقات التي وضحها المؤلف بين الطريقة الأولى والثانية؟"
Causal: "وفقاً للنص، ما الأسباب التي ذكرها الباحث وراء هذه النتيجة؟"
Evaluative: "بناءً على الأدلة المقدمة في النص، ما تقييم الكاتب للحل المقترح؟"
Synthesis: "كيف ربط النص بين المفهومين المذكورين في الفقرة الثانية؟"

AVOID questions like these that require external knowledge:

❌ "ما هي عاصمة فرنسا؟" (requires general knowledge)
❌ "متى حدثت الحرب العالمية الثانية؟" (historical fact not in text)
❌ "ما هي فوائد فيتامين سي بشكل عام؟" (general medical knowledge)

Do not use meta-referential phrases like "حسب النص" (according to the text), "ذكر الكاتب" (the author mentioned), "وفقاً للنص" (according to the text) - ask directly about the content

The response should be only the numbered list of questions in Arabic — no explanations, translations, or additional commentary."""
            
            # Create messages for the conversation
            messages = [
                {"role": "user", "content": f"{instruction}\n\nText: {context}"}
            ]

            logger.debug(f"Generating {self.num_questions_per_chunk} diverse questions from context...")
            
            # Generate response using pipeline
            outputs = self.pipeline(
                messages,
                max_new_tokens=2048,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample
            )
            
            # Extract the generated questions
            response_text = outputs[0]["generated_text"][-1]["content"].strip()
            
            # Parse the numbered list to extract individual questions
            questions = self._parse_numbered_questions(response_text)
            
            # Ensure we have the right number of questions
            if len(questions) < self.num_questions_per_chunk:
                logger.warning(f"Generated only {len(questions)} questions, expected {self.num_questions_per_chunk}")
            elif len(questions) > self.num_questions_per_chunk:
                questions = questions[:self.num_questions_per_chunk]
                logger.info(f"Truncated to {self.num_questions_per_chunk} questions")
            
            # Clean up each question
            cleaned_questions = []
            for question in questions:
                cleaned_question = self._clean_question(question)
                if cleaned_question:
                    cleaned_questions.append(cleaned_question)
            
            logger.debug(f"LLM generated {len(cleaned_questions)} diverse questions")
            for i, q in enumerate(cleaned_questions):
                logger.debug(f"Question {i+1}: {q[:100]}...")
            
            return cleaned_questions
            
        except Exception as e:
            logger.error(f"Error generating diverse questions from context with LLM: {e}")
            return []
    
    def _generate_all_questions_from_context_openai(self, context: str) -> List[str]:
        """Generate all questions for a chunk in one prompt using OpenAI."""
        try:
            instruction = f"""Based on the following Arabic text, generate exactly {self.num_questions_per_chunk} diverse, self-contained questions in Arabic. CRITICAL REQUIREMENT: Each question must be answerable ONLY using the information explicitly provided in the given text chunk - no external knowledge should be required.
The questions should:

Be answerable exclusively from the provided text - every piece of information needed to answer the question must be present in the text chunk
Be completely self-contained - include all relevant details, names, dates, facts, or context mentioned in the text
Be natural and conversational in Arabic
Cover various cognitive levels and question types:

Factual/Recall: Direct information retrieval from the text (ما، من، متى، أين)
Comprehension: Understanding concepts explained in the text (كيف، لماذا)
Application: Using information presented in the text (ماذا لو، كيف يمكن)
Analysis: Analyzing relationships and connections within the text (ما العلاقة بين، قارن)
Synthesis: Combining ideas presented in the text (ما الاستنتاج، اربط بين)
Evaluation: Making judgments based on information in the text (بناءً على النص، قيّم)


Never require external knowledge - avoid questions about general facts, historical events, or information not mentioned in the text
Focus only on content, concepts, examples, data, and information explicitly stated in the provided text
Be diverse and non-repetitive - each question should focus on different aspects, details, or perspectives from the text
Vary in complexity, depth, and scope while staying within the text's boundaries
Use natural Arabic phrasing and question structures

Format your response as a numbered list:

[First question in Arabic]
[Second question in Arabic]
[Third question in Arabic]
... and so on for {self.num_questions_per_chunk} questions

IMPORTANT: All questions must reference specific information from the text. Examples of context-dependent questions:

Factual: "ما هي المكونات الثلاثة التي ذكرها الكاتب في الوصفة؟"
Analytical: "حسب النص، كيف تؤثر درجة الحرارة المذكورة على عملية النمو؟"
Comparative: "ما الفروقات التي وضحها المؤلف بين الطريقة الأولى والثانية؟"
Causal: "وفقاً للنص، ما الأسباب التي ذكرها الباحث وراء هذه النتيجة؟"
Evaluative: "بناءً على الأدلة المقدمة في النص، ما تقييم الكاتب للحل المقترح؟"
Synthesis: "كيف ربط النص بين المفهومين المذكورين في الفقرة الثانية؟"

AVOID questions like these that require external knowledge:

❌ "ما هي عاصمة فرنسا؟" (requires general knowledge)
❌ "متى حدثت الحرب العالمية الثانية؟" (historical fact not in text)
❌ "ما هي فوائد فيتامين سي بشكل عام؟" (general medical knowledge)

Do not use meta-referential phrases like "حسب النص" (according to the text), "ذكر الكاتب" (the author mentioned), "وفقاً للنص" (according to the text) - ask directly about the content

The response should be only the numbered list of questions in Arabic — no explanations, translations, or additional commentary."""

            prompt = f"{instruction}\n\nText: {context}"
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=self.temperature,
                top_p=self.top_p,
                n=1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the numbered list to extract individual questions
            questions = self._parse_numbered_questions(response_text)
            
            # Ensure we have the right number of questions
            if len(questions) < self.num_questions_per_chunk:
                logger.warning(f"Generated only {len(questions)} questions, expected {self.num_questions_per_chunk}")
            elif len(questions) > self.num_questions_per_chunk:
                questions = questions[:self.num_questions_per_chunk]
                logger.info(f"Truncated to {self.num_questions_per_chunk} questions")
            
            # Clean up each question
            cleaned_questions = []
            for question in questions:
                cleaned_question = self._clean_question(question)
                if cleaned_question:
                    cleaned_questions.append(cleaned_question)
            
            logger.debug(f"OpenAI generated {len(cleaned_questions)} diverse questions")
            return cleaned_questions
            
        except Exception as e:
            logger.error(f"Error generating diverse questions from context with OpenAI: {e}")
            return []
    
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
            instruction = """Based on the following Arabic text, generate one question in Arabic that can **ONLY** be answered using the information explicitly provided in the given text.

**CRITICAL REQUIREMENT: The question must be answerable exclusively from the provided text chunk - no external knowledge should be required.**

The question should:
1. **Be answerable only from the provided text** - every piece of information needed to answer the question must be present in the text
2. Be natural and conversational in Arabic
3. Cover one of various cognitive types:
   - **Factual/Recall**: Direct information retrieval from the text (ما، من، متى، أين)
   - **Comprehension**: Understanding concepts explained in the text (كيف، لماذا)
   - **Application**: Using information presented in the text (ماذا لو، كيف يمكن)
   - **Analysis**: Analyzing relationships within the text (ما العلاقة بين، قارن)
   - **Synthesis**: Combining ideas presented in the text (ما الاستنتاج، اربط بين)
   - **Evaluation**: Making judgments based on information in the text (بناءً على النص، قيّم)
4. **Never require external knowledge** - avoid questions about general facts, historical events, or information not mentioned in the text
5. Focus only on content, concepts, examples, data, and information explicitly stated in the provided text
6. Include necessary context from the text within the question to make it clear and specific
7. **Do not use meta-referential phrases** like "حسب النص" (according to the text), "ذكر الكاتب" (the author mentioned), "وفقاً للنص" (according to the text) - ask directly about the content

**Examples of good context-dependent questions:**
- "ما هي الخطوات الثلاث لتحضير هذا الطبق؟" (What are the three steps for preparing this dish?)
- "كيف أثرت التغييرات المناخية على المحاصيل في هذه المنطقة؟" (How did climate changes affect crops in this region?)
- "ما الفرق بين الطريقتين المذكورتين؟" (What difference between the two mentioned methods?)

**AVOID questions like these that require external knowledge:**
- ❌ "ما هي عاصمة مصر؟" (requires general geographic knowledge)
- ❌ "متى تأسست الأمم المتحدة؟" (historical fact not in text)
- ❌ "ما هي خصائص الماء بشكل عام؟" (general scientific knowledge)

The response should be only the question in Arabic — no explanations, translations, or additional commentary."""
            
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
            instruction = """Based on the following Arabic text, provide a comprehensive and detailed answer to the given question in Arabic using ONLY the information explicitly provided in the text.
CRITICAL REQUIREMENT: The answer must be based exclusively on the provided text - do not add any external knowledge, general facts, or information not mentioned in the text.
The answer should:

Use only information from the provided text - every fact, detail, and explanation must come directly from the source text
Be thorough and well-structured based on available information in the text
Include all relevant details, examples, and explanations that are mentioned in the text
Be written in clear, natural Arabic
Provide a complete response using only what the text provides to address the question
Be educational and informative within the boundaries of the given text
Use proper Arabic grammar and style
If the text doesn't contain enough information to fully answer the question, acknowledge this limitation

Guidelines for context-based answers:

Quote or paraphrase specific information from the text
Organize the information logically as presented in the text
Never supplement with external knowledge even if it would make the answer more complete
If certain aspects of the question cannot be answered from the text, state this clearly

Example of a good context-based answer:
Question: "ما هي الخطوات المذكورة لتحضير الطبق؟"
Answer: "هناك ثلاث خطوات رئيسية لتحضير هذا الطبق: أولاً، تحضير المكونات التالية... ثانياً، عملية الخلط التي تتم عبر... وأخيراً، مرحلة الطبخ التي تستغرق..."
AVOID adding external information like:

❌ General facts not mentioned in the text
❌ Historical context not provided in the source
❌ Scientific explanations beyond what's in the text
❌ Personal opinions or interpretations not supported by the text

Focus on extracting, organizing, and presenting the information that is explicitly available in the provided text to create the most comprehensive answer possible within those boundaries."""
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
    
    def _parse_numbered_questions(self, response_text: str) -> List[str]:
        """Parse numbered questions from the LLM response."""
        questions = []
        
        # Split by lines and look for numbered patterns
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match patterns like "1.", "1-", "1)", "1-", etc.
            match = re.match(r'^\d+[\.\)\-\s]+(.+)$', line)
            if match:
                question = match.group(1).strip()
                if question:
                    questions.append(question)
        
        # If no numbered format found, try to split by other patterns
        if not questions:
            # Try splitting by question marks
            parts = response_text.split('؟')
            for part in parts:
                part = part.strip()
                if part and len(part) > 10:  # Minimum question length
                    # Add back the question mark
                    questions.append(part + '؟')
        
        return questions
    
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