import os
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from uuid import uuid4
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
from datetime import datetime

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize OpenAI Chat model
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Set page config early
st.set_page_config(layout="wide")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "files" not in st.session_state:
    st.session_state.files = {}

# Custom HTML layout
st.markdown("""
<style>
    .main-container {
        display: grid;
        grid-template-columns: 1fr 3fr;
        gap: 20px;
    }
    .sidebar-custom {
        background-color: #f9f9f9;
        padding: 15px;
        border-right: 1px solid #ddd;
        height: 100vh;
        overflow-y: auto;
    }
    .chat-container {
        padding: 20px;
    }
    .chat-input {
        margin-top: 20px;
    }
    .graph-section {
        margin-top: 30px;
    }
</style>
<div class="main-container">
    <div class="sidebar-custom">
        <h4>Chat History</h4>
        <div id="chat-history">
""", unsafe_allow_html=True)

for i, (q, a, doc) in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**Q{i+1}: {q} ({doc})**<br>{a}<br><br>", unsafe_allow_html=True)

st.markdown("""
        </div>
    </div>
    <div class="chat-container">
""", unsafe_allow_html=True)

question = st.text_input("Ask a question about your documents:", key="question_input")
uploaded_files = st.file_uploader("Upload PDFs, Word Docs, or Images", type=["pdf", "docx", "png", "jpg"], accept_multiple_files=True, key="uploader")

for i, file in enumerate(uploaded_files or []):
    file_id = str(uuid4())[:8]
    block_label = f"Block {chr(65 + i)}"
    st.session_state.files[block_label] = file_id

    file_type = file.name.split(".")[-1].lower()
    text = ""

    try:
        if file_type == "pdf":
            text = "\n".join([page.extract_text() for page in PdfReader(file).pages if page.extract_text()])
        elif file_type == "docx":
            text = "\n".join([para.text for para in DocxDocument(file).paragraphs])
        elif file_type in ["png", "jpg"]:
            text = pytesseract.image_to_string(Image.open(file))
    except Exception as e:
        st.warning(f"Failed to process {file.name}: {e}")

    if text:
        prompt = f"""
        Extract key entities and relationships from the following content as triples (subject - relation - object):\n{text}\n
        Return the output in a list of triples.
        """
        response = chat([HumanMessage(content=prompt)])
        triples = response.content
        with driver.session() as session:
            for triple in triples.split("\n"):
                if '-' in triple:
                    parts = triple.split('-')
                    if len(parts) == 3:
                        subject, relation, obj = [p.strip() for p in parts]
                        session.run("""
                            MERGE (s:Entity {name: $subject, file_id: $file_id})
                            MERGE (o:Entity {name: $object, file_id: $file_id})
                            MERGE (s)-[r:RELATION {type: $relation, file_id: $file_id}]->(o)
                        """, subject=subject, object=obj, relation=relation, file_id=file_id)

selected_block = st.selectbox("Select Document Block", list(st.session_state.files.keys()) if st.session_state.files else [], key="doc_selector")
if question and selected_block:
    file_id = st.session_state.files[selected_block]
    cypher_prompt = f"""
    Given the user question: \"{question}\", write a Cypher query that could help answer it using a graph of entities and relationships. Assume the graph contains nodes with 'name' and 'file_id'.
    """
    cypher = chat([HumanMessage(content=cypher_prompt)]).content

    with driver.session() as session:
        results = session.run(cypher)
        extracted_info = "\n".join([str(record.data()) for record in results])

    answer_prompt = f"""
    Based on the following graph results, answer the question: \"{question}\"
    Graph Data:\n{extracted_info}
    """
    answer = chat([HumanMessage(content=answer_prompt)]).content

    st.session_state.history.append((question, answer, selected_block))

    st.markdown(f"""
    <div class='chat-input'><h4>Answer</h4><p>{answer}</p></div>
    <div><strong>Generated Cypher Query:</strong><pre>{cypher}</pre></div>
    """, unsafe_allow_html=True)

if selected_block:
    file_id = st.session_state.files[selected_block]
    G = nx.DiGraph()
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Entity {file_id: $file_id})-[r:RELATION]->(b:Entity {file_id: $file_id})
            RETURN a.name AS from, r.type AS rel, b.name AS to
        """, file_id=file_id)
        for record in result:
            G.add_node(record["from"])
            G.add_node(record["to"])
            G.add_edge(record["from"], record["to"], label=record["rel"])

    net = Network(height="500px", width="100%", directed=True)
    net.from_nx(G)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.show(tmp_file.name)
    graph_html = open(tmp_file.name, "r", encoding="utf-8").read()

    st.markdown("""<div class="graph-section"><h4>Knowledge Graph</h4></div>""", unsafe_allow_html=True)
    components.html(graph_html, height=550)
    st.download_button(
        label="Download Graph as HTML",
        data=open(tmp_file.name, "rb").read(),
        file_name=f"graph_{file_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.html",
        mime="text/html"
    )

st.markdown("""
    </div> <!-- End chat container -->
</div> <!-- End main container -->
""", unsafe_allow_html=True)
