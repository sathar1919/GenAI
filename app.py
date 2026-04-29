"""
Research Co-Pilot - Main Streamlit Application
"""

# ============================================
# STEP 1: Import streamlit and set page config (MUST BE FIRST)
# ============================================
import streamlit as st
st.set_page_config(
    page_title="Research Co-Pilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# STEP 2: All other imports
# ============================================
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env with error handling
try:
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path, encoding='utf-8')
except Exception as e:
    st.warning(f"⚠️ Could not load .env file: {e}")

# Import custom modules
from src.utils.pdf_processor import PDFProcessor
from src.utils.vector_store import ResearchVectorStore
from src.agents.research_agent import ResearchAgent
from src.agents.literature_agent import LiteratureReviewAgent

# ============================================
# STEP 3: Custom CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E5A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# STEP 4: Sidebar content & API Key Handling
# ============================================
# Change this part in your sidebar:
with st.sidebar:
    st.markdown("## 🎯 Research Co-Pilot")
    st.markdown("---")
    
    # Updated to ask for Groq Key
    groq_key = st.text_input("Groq API Key", type="password", 
                             help="Enter your Groq API key for Llama 3.3")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.session_state.research_agent = ResearchAgent()
        st.session_state.literature_agent = LiteratureReviewAgent()
        st.success("✅ Groq API Key set and Agents Updated!")
    
    st.markdown("---")
    st.markdown("### Features")
    st.markdown("""
    - 📄 Paper Summarization
    - 🔍 Research Gap Analysis
    - 📚 Literature Review
    - 📝 Citation Generation
    - 🔎 Semantic Search
    """)

# ============================================
# STEP 5: Initialize session state
# ============================================
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = ResearchVectorStore()
if 'research_agent' not in st.session_state:
    st.session_state.research_agent = ResearchAgent()
if 'literature_agent' not in st.session_state:
    st.session_state.literature_agent = LiteratureReviewAgent()
if 'processed_papers' not in st.session_state:
    st.session_state.processed_papers = []

# ============================================
# STEP 6: Main content
# ============================================
st.markdown('<div class="main-header">📚 LLM-Powered Research Co-Pilot</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Paper Analysis", 
    "🔍 Research Gaps", 
    "📚 Literature Review",
    "🔎 Search Papers"
])

with tab1:
    st.markdown('<div class="sub-header">📄 Upload & Analyze Research Papers</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=['pdf'])
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            if st.button("📊 Process Paper", type="primary"):
                with st.spinner("Processing paper..."):
                    processor = PDFProcessor()
                    paper_text = processor.extract_text(tmp_path)
                    sections = processor.extract_sections(tmp_path)
                    
                    metadata = {"filename": uploaded_file.name, "sections": list(sections.keys())}
                    st.session_state.vector_store.add_paper(paper_text, metadata)
                    
                    summary = st.session_state.research_agent.summarize_paper(paper_text)
                    keywords = st.session_state.research_agent.extract_keywords(paper_text)
                    
                    st.session_state.processed_papers.append({
                        "name": uploaded_file.name,
                        "summary": summary,
                        "keywords": keywords,
                        "sections": sections
                    })
                    
                    st.success(f"✅ Successfully processed: {uploaded_file.name}")
                    
                    st.markdown("### 📝 Paper Summary")
                    st.write(summary)
                    
                    st.markdown("### 🏷️ Key Keywords")
                    st.write(", ".join(keywords))
                    
            os.unlink(tmp_path)
    
    with col2:
        st.markdown("### 📊 Processed Papers")
        if st.session_state.processed_papers:
            for paper in st.session_state.processed_papers:
                st.info(f"✅ {paper['name']}")
        else:
            st.info("No papers processed yet")

with tab2:
    st.markdown('<div class="sub-header">🔍 Research Gap Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_papers:
        selected_paper = st.selectbox("Select a paper", [p['name'] for p in st.session_state.processed_papers])
        
        if st.button("🔍 Identify Research Gaps", type="primary"):
            with st.spinner("Analyzing research gaps..."):
                paper_data = next(p for p in st.session_state.processed_papers if p['name'] == selected_paper)
                keywords_str = ", ".join(paper_data['keywords'])
                related_papers = st.session_state.literature_agent.search_papers(keywords_str, max_results=3)
                related_context = "\n".join([f"{p['title']}: {p['abstract'][:300]}" for p in related_papers])
                
                gaps = st.session_state.research_agent.identify_research_gaps(
                    paper_data['sections'].get('conclusion', ''), related_context
                )
                
                st.markdown("### 🎯 Identified Research Gaps")
                st.markdown('<div class="info-box">' + gaps + '</div>', unsafe_allow_html=True)
    else:
        st.warning("Please upload and process a paper first in the Paper Analysis tab")

with tab3:
    st.markdown('<div class="sub-header">📚 Systematic Literature Review</div>', unsafe_allow_html=True)
    research_topic = st.text_input("Enter research topic", placeholder="e.g., Transformer models")
    num_papers = st.slider("Number of papers to review", 3, 15, 5)
    
    if st.button("📚 Conduct Literature Review", type="primary") and research_topic:
        with st.spinner("Searching and analyzing..."):
            papers = st.session_state.literature_agent.search_papers(research_topic, num_papers)
            review = st.session_state.literature_agent.conduct_literature_review(research_topic, papers)
            
            st.markdown("### 📝 Literature Review Results")
            st.markdown('<div class="success-box">' + review + '</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="sub-header">🔎 Semantic Search</div>', unsafe_allow_html=True)
    search_query = st.text_input("Ask a question about your papers")
    
    if st.button("🔎 Search", type="primary") and search_query:
        with st.spinner("Searching..."):
            results = st.session_state.vector_store.search(search_query, k=5)
            if results:
                for i, result in enumerate(results):
                    st.markdown(f"**Result {i+1}**")
                    st.markdown(f"📄 {result['content'][:500]}...")
                    st.markdown("---")
            else:
                st.info("No results found.")