"""
Literature Review Agent for systematic review
"""

import arxiv
import os
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

class LiteratureReviewAgent:
    """Agent for conducting systematic literature reviews"""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        # Updated to use Groq and Llama 3.3
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            temperature=0.2,
            groq_api_key=api_key
        ) if api_key else None
        
        self.arxiv_client = arxiv.Client()
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for papers on ArXiv"""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in self.arxiv_client.results(search):
            papers.append({
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "pdf_url": result.pdf_url,
                "published": result.published,
                "categories": result.categories
            })
        
        return papers
    
    def conduct_literature_review(self, topic: str, papers: List[Dict]) -> str:
        """Conduct systematic literature review on given papers"""
        if not self.llm:
            return "[DEMO MODE] Please set your GROQ_API_KEY to conduct a full review."
            
        paper_summaries = []
        for i, paper in enumerate(papers[:5]):  
            summary = f"Paper {i+1}: {paper['title']}\nAbstract: {paper['abstract'][:500]}\n"
            paper_summaries.append(summary)
        
        combined_papers = "\n---\n".join(paper_summaries)
        
        review_prompt = PromptTemplate(
            input_variables=["topic", "papers"],
            template="""
            Conduct a systematic literature review on the topic: "{topic}"
            
            Based on these research papers:
            {papers}
            
            Provide a structured review including:
            1. Overview of current research landscape
            2. Common methodologies used
            3. Key findings across studies
            4. Identified research gaps
            5. Recommendations for future research
            
            Literature Review:
            """
        )
        
        # LCEL Syntax
        chain = review_prompt | self.llm
        response = chain.invoke({"topic": topic, "papers": combined_papers})
        return response.content
    
    def synthesize_findings(self, papers: List[Dict]) -> Dict:
        """Synthesize key findings from multiple papers"""
        themes = {}
        
        for paper in papers:
            abstract = paper.get('abstract', '')
            if 'machine learning' in abstract.lower():
                themes['ML Approach'] = themes.get('ML Approach', []) + [paper['title']]
            if 'deep learning' in abstract.lower():
                themes['Deep Learning'] = themes.get('Deep Learning', []) + [paper['title']]
            if 'nlp' in abstract.lower() or 'natural language' in abstract.lower():
                themes['NLP'] = themes.get('NLP', []) + [paper['title']]
        
        return themes