# app.py (without database dashboard)
import streamlit as st
import sqlite3
import re
import ast
from dotenv import load_dotenv
import os
import base64
from gtts import gTTS
import tempfile

from langchain_groq import ChatGroq
from langchain.chains import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool

# Function to convert text to speech and create a playable audio element
def text_to_speech(text):
    """Convert text to speech and return an HTML audio player"""
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            # Generate the speech using gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_audio.name)
            
            # Read the audio file and encode as base64
            with open(temp_audio.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Create a data URL for the audio
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f'''
            <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            '''
            return audio_html
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# Page configuration
st.set_page_config(page_title="Hospital Database NLP Query System", layout="wide")

# Title and description
st.title("Hospital Database NLP Query System")
st.markdown("""
Enter natural language questions about the hospital database and get answers instantly.
Examples: "Which doctors treat flu?", "Show me all patients over 40", etc.
""")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Connect to database
@st.cache_resource
def get_db():
    db_path = "sqlite:///Hospital.db"
    return SQLDatabase.from_uri(db_path)

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key
    )

# Initialize database and LLM
db = get_db()
llm = get_llm()

# Create SQL query chain
@st.cache_resource
def get_chains():
    sql_query_chain = create_sql_query_chain(llm=llm, db=db)
    query_execute = QuerySQLDatabaseTool(db=db)
    return sql_query_chain, query_execute

sql_query_chain, query_execute = get_chains()

# Enable voice output toggle in sidebar
with st.sidebar:
    st.subheader("Settings")
    enable_voice = st.toggle("Enable Voice Output", value=False)

# User input
user_question = st.text_input("Enter your question about the Hospital database:", 
                            placeholder="e.g., Which doctors are assigned to treat Hypertension?")

# Process button
if st.button("Get Answer") or user_question:
    if user_question:
        with st.spinner("Processing your question..."):
            try:
                # Get the query from the chain
                full_response = sql_query_chain.invoke({"question": user_question})
                
                # Display generated SQL (collapsible)
                with st.expander("View Generated SQL"):
                    st.code(full_response, language="sql")
                
                # Extract just the SQL part
                sql_only = re.search(r'SQLQuery: (.*)', full_response, re.DOTALL)
                if sql_only:
                    clean_query = sql_only.group(1).strip()
                    
                    # Execute the SQL
                    raw_result = query_execute.invoke(clean_query)
                    
                    # Format and display the results
                    st.subheader("Results")
                    try:
                        # Parse the results
                        if isinstance(raw_result, list):
                            results = raw_result
                        else:
                            results = ast.literal_eval(raw_result)
                        
                        if len(results) == 0:
                            st.info("No results found.")
                        else:
                            st.table(results)
                            
                            # Generate natural language response
                            st.subheader("Summary")
                            nl_prompt = f"""
                            Based on the query "{user_question}" and the results {results},
                            provide a brief, clear summary of the findings in natural language.
                            """
                            nl_response = llm.invoke(nl_prompt).content
                            st.write(nl_response)
                            
                            # Voice output if enabled
                            if enable_voice:
                                st.subheader("Voice Summary")
                                st.markdown("Listen to the summary:")
                                audio_html = text_to_speech(nl_response)
                                if audio_html:
                                    st.markdown(audio_html, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error formatting results: {e}")
                        st.write(raw_result)
                else:
                    st.error("Could not extract SQL query from response")
            except Exception as e:
                st.error(f"Error processing question: {e}")
    else:
        st.warning("Please enter a question.")

# Database dashboard removed from sidebar