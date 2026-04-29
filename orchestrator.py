"""
Multi-Agent Orchestration using CrewAI
"""

from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from typing import List, Dict
import os

class ResearchOrchestrator:
    """Orchestrates multiple research agents using CrewAI"""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        # Standardized to Groq
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            temperature=0.3,
            groq_api_key=api_key
        ) if api_key else None
        
        if self.llm:
            self.researcher_agent = Agent(
                role='Senior Research Analyst',
                goal='Conduct thorough analysis of research papers',
                backstory="Expert in academic research with 10+ years experience",
                llm=self.llm,
                verbose=True
            )
            
            self.summarizer_agent = Agent(
                role='Research Summarizer',
                goal='Create concise, informative summaries',
                backstory="Specialist in extracting key insights from complex papers",
                llm=self.llm,
                verbose=True
            )
            
            self.gap_analyst_agent = Agent(
                role='Research Gap Analyst',
                goal='Identify novel research opportunities',
                backstory="Expert at finding unexplored areas in research",
                llm=self.llm,
                verbose=True
            )
    
    def analyze_paper_complete(self, paper_text: str) -> Dict:
        """Run complete analysis pipeline using multiple agents"""
        if not self.llm:
            return {"analysis": "Please set GROQ_API_KEY to use CrewAI"}

        analysis_task = Task(
            description=f"Analyze this research paper: {paper_text[:5000]}",
            agent=self.researcher_agent,
            expected_output="Detailed analysis of methodology and findings"
        )
        
        summary_task = Task(
            description="Create a comprehensive summary of the paper",
            agent=self.summarizer_agent,
            expected_output="Structured summary with key points"
        )
        
        gap_task = Task(
            description="Identify research gaps and future directions",
            agent=self.gap_analyst_agent,
            expected_output="List of research gaps and recommendations"
        )
        
        crew = Crew(
            agents=[self.researcher_agent, self.summarizer_agent, self.gap_analyst_agent],
            tasks=[analysis_task, summary_task, gap_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return {"analysis": result}