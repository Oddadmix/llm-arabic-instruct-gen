#!/usr/bin/env python3
"""
Test script for the updated QA generator with two-step process.
"""

import logging
from processors.qa_generator import QAGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qa_generator():
    """Test the QA generator with fallback methods."""
    
    # Initialize QA generator with offloading to avoid loading the model
    qa_generator = QAGenerator(
        num_questions_per_chunk=2,
        llm_model="Qwen/Qwen3-8B",
        offload_model=True  # This will prevent model loading for testing
    )
    
    # Test text chunk
    test_chunk = """
    الذكاء الاصطناعي هو مجال من مجالات علوم الحاسوب يهدف إلى إنشاء أنظمة قادرة على أداء مهام تتطلب عادة ذكاءً بشرياً. 
    تشمل هذه المهام التعلم والاستنتاج وحل المشكلات والإدراك واللغة الطبيعية. 
    يشمل الذكاء الاصطناعي عدة تقنيات مثل التعلم الآلي والشبكات العصبية ومعالجة اللغة الطبيعية.
    """
    
    logger.info("Testing QA generation with fallback methods...")
    
    # Test question generation
    logger.info("Testing question generation...")
    question = qa_generator._generate_fallback_question(test_chunk, 0)
    logger.info(f"Generated question: {question}")
    
    # Test answer generation
    logger.info("Testing answer generation...")
    answer = qa_generator._generate_fallback_answer(test_chunk, question)
    logger.info(f"Generated answer: {answer}")
    
    # Test full QA pair generation
    logger.info("Testing full QA pair generation...")
    qa_pairs = qa_generator._generate_qa_for_chunk(test_chunk, 0)
    
    logger.info(f"Generated {len(qa_pairs)} QA pairs:")
    for i, qa_pair in enumerate(qa_pairs):
        logger.info(f"QA Pair {i+1}:")
        logger.info(f"  Question: {qa_pair['question']}")
        logger.info(f"  Answer: {qa_pair['answer']}")
        logger.info(f"  ID: {qa_pair['id']}")
        logger.info("---")
    
    logger.info("Test completed successfully!")

if __name__ == "__main__":
    test_qa_generator() 