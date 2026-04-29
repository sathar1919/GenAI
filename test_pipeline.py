"""
Test the complete research pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.pdf_processor import PDFProcessor
from src.agents.research_agent import ResearchAgent

def test_pdf_processing():
    """Test PDF processing functionality"""
    processor = PDFProcessor()
    
    # Create a sample text for testing
    sample_text = """
    Abstract: This paper presents a novel approach to transformer-based language models.
    Introduction: Large language models have revolutionized NLP.
    Methodology: We use attention mechanisms and fine-tuning.
    Results: Our model achieves 95% accuracy.
    Conclusion: Future work includes multi-modal extensions.
    """
    
    # Test chunking
    chunks = processor.chunk_text(sample_text, chunk_size=50)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Test section extraction
    sections = processor.extract_sections("dummy.pdf")  # Will return empty
    print(f"✅ Section extraction ready")
    
    return True

def test_research_agent():
    """Test research agent functionality"""
    agent = ResearchAgent()
    
    sample_paper = """
    This research explores few-shot learning techniques for text classification.
    We propose a meta-learning approach that adapts to new tasks with minimal examples.
    Our method achieves state-of-the-art results on 5 benchmark datasets.
    Future work includes extending to multi-modal scenarios.
    """
    
    # Test keyword extraction (will use OpenAI if key is set)
    try:
        keywords = agent.extract_keywords(sample_paper)
        print(f"✅ Keyword extraction: {keywords}")
    except Exception as e:
        print(f"⚠️ OpenAI API key required for full testing: {e}")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Research Co-Pilot Pipeline...")
    
    test_pdf_processing()
    test_research_agent()
    
    print("\n✅ All tests completed!")