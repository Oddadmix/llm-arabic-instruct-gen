#!/usr/bin/env python3
"""
Demonstration script showing the difference between individual and batch question generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.qa_generator import QAGenerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_batch_vs_individual():
    """Demonstrate the difference between batch and individual question generation."""
    
    # Sample Arabic text for demonstration (contains specific information)
    sample_text = """
    وصفة الكنافة التقليدية تتطلب ثلاثة مكونات أساسية: عجينة الكنافة، الجبنة البيضاء، والسمن البلدي. 
    طريقة التحضير تبدأ بتسخين السمن في مقلاة على نار متوسطة لمدة خمس دقائق، ثم إضافة عجينة الكنافة وتقليبها حتى تصبح ذهبية اللون.
    بعد ذلك، يتم إضافة الجبنة البيضاء بين طبقتين من عجينة الكنافة، ثم رش القطر الساخن فوقها.
    درجة الحرارة المثالية لتحضير الكنافة هي 180 درجة مئوية، وتحتاج إلى 15 دقيقة في الفرن.
    """
    
    logger.info("=== Batch vs Individual Question Generation Demo ===\n")
    
    try:
        # Initialize QA generator
        qa_generator = QAGenerator(
            num_questions_per_chunk=3,
            llm_model="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            offload_model=True,
            llm_backend="transformers"
        )
        
        logger.info("✅ QA Generator initialized successfully\n")
        
        # Demonstrate batch question generation (NEW APPROACH)
        logger.info("🔄 NEW APPROACH: Batch Question Generation")
        logger.info("Generating all questions for the chunk in one prompt...")
        
        questions_batch = qa_generator._generate_all_questions_from_context_transformers(sample_text)
        
        logger.info(f"Generated {len(questions_batch)} questions using batch method:")
        for i, question in enumerate(questions_batch, 1):
            logger.info(f"  {i}. {question}")
        
        logger.info("\n" + "="*60 + "\n")
        
        # Demonstrate individual answer generation
        logger.info("📝 Generating answers individually for each question...")
        qa_pairs = []
        
        for i, question in enumerate(questions_batch):
            logger.info(f"Generating answer for question {i+1}...")
            answer = qa_generator._generate_answer_from_question_context_transformers(sample_text, question)
            qa_pairs.append({
                "question": question,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer
            })
        
        logger.info("\n📊 Final QA Pairs Generated:")
        for i, qa in enumerate(qa_pairs, 1):
            logger.info(f"\n--- QA Pair {i} ---")
            logger.info(f"Question: {qa['question']}")
            logger.info(f"Answer: {qa['answer']}")
        
        logger.info("\n" + "="*60)
        logger.info("🎯 BENEFITS OF BATCH QUESTION GENERATION:")
        logger.info("✅ Ensures question diversity - no repetitive questions")
        logger.info("✅ Context-aware - model sees full chunk when generating all questions")
        logger.info("✅ Efficient - fewer LLM calls for question generation")
        logger.info("✅ Quality control - model can ensure coverage of different aspects")
        logger.info("✅ Individual answers - each answer gets full context attention")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Batch QA Generation Demo...")
    success = demonstrate_batch_vs_individual()
    if success:
        print("\n🎉 Demo completed successfully!")
    else:
        print("\n💥 Demo failed!")
        sys.exit(1) 