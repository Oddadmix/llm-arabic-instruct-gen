#!/usr/bin/env python3
"""
Test script for text-only question generation feature.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processors.qa_generator import QAGenerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_text_only_qa_generation():
    """Test the new text-only question generation feature."""
    
    # Sample Arabic text for testing (contains specific information)
    sample_text = """
    وصفة الكنافة التقليدية تتطلب ثلاثة مكونات أساسية: عجينة الكنافة، الجبنة البيضاء، والسمن البلدي. 
    طريقة التحضير تبدأ بتسخين السمن في مقلاة على نار متوسطة لمدة خمس دقائق، ثم إضافة عجينة الكنافة وتقليبها حتى تصبح ذهبية اللون.
    بعد ذلك، يتم إضافة الجبنة البيضاء بين طبقتين من عجينة الكنافة، ثم رش القطر الساخن فوقها.
    درجة الحرارة المثالية لتحضير الكنافة هي 180 درجة مئوية، وتحتاج إلى 15 دقيقة في الفرن.
    """
    
    logger.info("Testing text-only question generation...")
    
    try:
        # Initialize QA generator with small model for testing
        qa_generator = QAGenerator(
            num_questions_per_chunk=3,
            llm_model="Qwen/Qwen2.5-7B-Instruct",  # Use smaller model for testing
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            offload_model=True,
            llm_backend="transformers"
        )
        
        logger.info("QA Generator initialized successfully")
        
        # Test batch question generation with text-only approach
        logger.info("Generating text-only questions using batch method...")
        questions = qa_generator._generate_all_questions_from_context_transformers(sample_text)
        
        logger.info(f"Generated {len(questions)} text-only questions:")
        for i, question in enumerate(questions, 1):
            logger.info(f"  {i}. {question}")
        
        # Test individual answer generation for each question
        logger.info("\nGenerating answers for each question...")
        for i, question in enumerate(questions, 1):
            logger.info(f"Generating answer for question {i}...")
            answer = qa_generator._generate_answer_from_question_context_transformers(sample_text, question)
            logger.info(f"Answer {i}: {answer[:150]}...")
        
        # Verify that questions are text-specific
        logger.info("\n" + "="*60)
        logger.info("🔍 VERIFICATION: Checking if questions are text-specific...")
        
        text_keywords = ["كنافة", "مكونات", "طريقة", "درجة الحرارة", "فرن", "جبنة", "سمن", "عجينة"]
        
        for i, question in enumerate(questions, 1):
            has_text_reference = any(keyword in question for keyword in text_keywords)
            logger.info(f"Question {i}: {'✅ Text-specific' if has_text_reference else '❌ May require external knowledge'}")
            logger.info(f"  Question: {question}")
        
        logger.info("Text-only question generation test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_text_only_qa_generation()
    if success:
        print("\n✅ Text-only question generation test passed!")
    else:
        print("\n❌ Text-only question generation test failed!")
        sys.exit(1) 