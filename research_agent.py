"""
Research Agent using LangChain for paper analysis
"""

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from typing import List, Dict
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path, encoding='utf-8')

class ResearchAgent:
    """AI Agent for research paper analysis and summarization"""
    
    def __init__(self, use_local_llm: bool = False):
        """Initialize the Research Agent"""
        
        # Get Groq API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        
        try:
            if api_key:
                # Updated to use Groq and Llama 3.3
                self.llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile", 
                    temperature=0.3,
                    groq_api_key=api_key
                )
                print("✅ Groq client initialized successfully")
            else:
                print("⚠️ Warning: Valid Groq API key not found. Using mock responses.")
                self.llm = None
        except Exception as e:
            print(f"⚠️ Error initializing Groq: {e}. Using mock responses.")
            self.llm = None
        
        # Initialize prompts
        self.summary_prompt = PromptTemplate(
            input_variables=["paper_text"],
            template="""
            You are a research assistant. Summarize the following research paper in 3-4 paragraphs.
            Include:
            1. Main research problem and objectives
            2. Methodology used
            3. Key findings and results
            4. Limitations and future work
            
            Paper Content:
            {paper_text}
            
            Summary:
            """
        )
        
        self.research_gap_prompt = PromptTemplate(
            input_variables=["paper_text", "related_papers"],
            template="""
            Based on the main paper and related research, identify research gaps:
            
            Main Paper: {paper_text}
            
            Related Papers Context: {related_papers}
            
            Identify:
            1. Unexplored areas in this research
            2. Limitations of current approaches
            3. Potential future research directions
            4. Novel problems that could be addressed
            
            Research Gaps:
            """
        )
        
        self.citation_prompt = PromptTemplate(
            input_variables=["paper_metadata", "style"],
            template="""
            Generate citations for this paper in {style} format.
            
            Paper Metadata: {paper_metadata}
            
            Citation:
            """
        )
        
        # Create LCEL Chains
        if self.llm:
            self.summary_chain = self.summary_prompt | self.llm
            self.gap_chain = self.research_gap_prompt | self.llm
            self.citation_chain = self.citation_prompt | self.llm
    
    def summarize_paper(self, paper_text: str) -> str:
        """Generate comprehensive summary of research paper"""
        if not self.llm:
            return self._mock_summary(paper_text)
        
        truncated_text = paper_text[:8000] if len(paper_text) > 8000 else paper_text
        response = self.summary_chain.invoke({"paper_text": truncated_text})
        return response.content
    
    def _mock_summary(self, paper_text: str) -> str:
        """Provide mock summary when no API key is available"""
        return """
        [DEMO MODE - No API Key Detected]
        
        To use the full AI features, please:
        1. Get a Groq API key from console.groq.com
        2. Add it to your sidebar or .env file as: GROQ_API_KEY=your_key_here
        """
    
    def identify_research_gaps(self, paper_text: str, related_context: str = "") -> str:
        """Identify research gaps from the paper"""
        if not self.llm:
            return "[DEMO MODE] Research gaps would be identified here. Please configure your Groq API key."
        
        truncated_text = paper_text[:6000] if len(paper_text) > 6000 else paper_text
        response = self.gap_chain.invoke({
            "paper_text": truncated_text,
            "related_papers": related_context if related_context else "No related papers provided."
        })
        return response.content
    
    def generate_citation(self, metadata: Dict, style: str = "APA") -> str:
        """Generate citation in specified format"""
        if not self.llm:
            return f"[DEMO MODE] Citation in {style} format would be generated here"
        
        response = self.citation_chain.invoke({"paper_metadata": str(metadata), "style": style})
        return response.content
    
    def extract_keywords(self, paper_text: str) -> List[str]:
        """Extract key terms and concepts from paper"""
        if not self.llm:
            return ["Machine Learning", "Deep Learning", "NLP", "Transformers", "LLM"]
        
        keyword_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract 5-7 key technical keywords from this research paper.
            Return as comma-separated list.
            
            Paper: {text}
            
            Keywords:
            """
        )
        chain = keyword_prompt | self.llm
        response = chain.invoke({"text": paper_text[:3000]})
        keywords_str = response.content
        return [k.strip() for k in keywords_str.split(",")]