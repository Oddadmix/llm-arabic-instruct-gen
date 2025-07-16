#!/usr/bin/env python3
"""
Test script for batch question generation feature.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processors.qa_generator import QAGenerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_batch_question_generation():
    """Test the new batch question generation feature."""
    
    # Sample Arabic text for testing
    sample_text = """
    القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها، وتعتبر أكبر مدينة عربية من حيث عدد السكان والمساحة، وتحتل المركز الثاني أفريقياً والسابع عشر عالمياً من حيث التعداد السكاني، يبلغ عدد سكانها 9.5 مليون نسمة حسب إحصائيات عام 2017، يمثلون 10.6% من إجمالي تعداد سكان مصر.
    
    تقع القاهرة على جوانب جزر نهر النيل في شمال مصر، مباشرة جنوب شرق النقطة التي يترك فيها نهر النيل واديه محصوراً في الصحراء منقسماً إلى فرعين داخل منطقة دلتا النيل، فرع رشيد وفرع دمياط.
    
    تأسست مدينة القاهرة في عهد الدولة الفاطمية، وأصبحت عاصمة لمصر منذ ذلك الحين، وتعتبر من أقدم المدن الإسلامية في العالم، حيث تضم العديد من المساجد التاريخية والقلاع والقصور القديمة.
    """
    
    logger.info("Testing batch question generation...")
    
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
        
        # Test batch question generation
        logger.info("Generating questions using batch method...")
        questions = qa_generator._generate_all_questions_from_context_transformers(sample_text)
        
        logger.info(f"Generated {len(questions)} questions:")
        for i, question in enumerate(questions, 1):
            logger.info(f"  {i}. {question}")
        
        # Test individual answer generation for each question
        logger.info("\nGenerating answers for each question...")
        for i, question in enumerate(questions, 1):
            logger.info(f"Generating answer for question {i}...")
            answer = qa_generator._generate_answer_from_question_context_transformers(sample_text, question)
            logger.info(f"Answer {i}: {answer[:100]}...")
        
        logger.info("Batch question generation test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_batch_question_generation()
    if success:
        print("\n✅ Batch question generation test passed!")
    else:
        print("\n❌ Batch question generation test failed!")
        sys.exit(1) 