"""
Run script for Research Co-Pilot
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import streamlit
        import langchain
        import PyPDF2
        print("✅ All dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def main():
    print("🚀 Starting Research Co-Pilot...")
    
    if not check_dependencies():
        sys.exit(1)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not set in environment")
        print("You can enter it in the sidebar of the application")
    
    # Run streamlit app
    print("📚 Launching application at http://localhost:8501")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()