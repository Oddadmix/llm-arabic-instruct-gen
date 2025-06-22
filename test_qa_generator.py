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
    """Test the QA generator initialization and structure."""
    
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
    
    logger.info("Testing QA generator initialization...")
    logger.info(f"QA Generator initialized with {qa_generator.num_questions_per_chunk} questions per chunk")
    logger.info(f"LLM Model: {qa_generator.llm_model}")
    logger.info(f"Offload model: {qa_generator.offload_model}")
    
    # Test that the generator structure is correct
    logger.info("Testing QA generator structure...")
    
    # Test that the methods exist
    assert hasattr(qa_generator, '_generate_question_from_context')
    assert hasattr(qa_generator, '_generate_answer_from_question_context')
    assert hasattr(qa_generator, '_clean_question')
    assert hasattr(qa_generator, '_clean_answer')
    
    logger.info("All required methods are present")
    
    # Test text cleaning methods
    logger.info("Testing text cleaning methods...")
    
    test_question = "ما هو الذكاء الاصطناعي"
    cleaned_question = qa_generator._clean_question(test_question)
    logger.info(f"Original question: {test_question}")
    logger.info(f"Cleaned question: {cleaned_question}")
    
    test_answer = "الذكاء الاصطناعي هو مجال من مجالات علوم الحاسوب"
    cleaned_answer = qa_generator._clean_answer(test_answer)
    logger.info(f"Original answer: {test_answer}")
    logger.info(f"Cleaned answer: {cleaned_answer}")
    
    logger.info("Test completed successfully!")
    logger.info("Note: LLM-based QA generation requires the model to be loaded and will be tested during actual processing")

if __name__ == "__main__":
    test_qa_generator() 