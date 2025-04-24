import os
import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Document Knowledge Graph RAG")

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
import re
import nltk
from nltk.tokenize import sent_tokenize
import textwrap
import hashlib

# UI Layout
st.title("Document Knowledge Graph RAG")
st.sidebar.title("Chat History")

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Neo4j driver
connection_status = st.sidebar.empty()
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Test connection
    with driver.session() as session:
        session.run("RETURN 1")
    connection_status.success("Connected to Neo4j")
except Exception as e:
    connection_status.error(f"Failed to connect to Neo4j: {str(e)}")
    driver = None

# Initialize OpenAI Chat model
openai_status = st.sidebar.empty()
try:
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    openai_status.success("Connected to OpenAI")
except Exception as e:
    openai_status.error(f"Failed to connect to OpenAI: {str(e)}")
    chat = None

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "file_ids" not in st.session_state:
    st.session_state.file_ids = []
if "file_names" not in st.session_state:
    st.session_state.file_names = {}
if "file_hashes" not in st.session_state:
    st.session_state.file_hashes = {}  # Store document hashes for duplicate detection
if "file_metadata" not in st.session_state:
    st.session_state.file_metadata = {}  # Store metadata about documents
if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None
if "chunks" not in st.session_state:
    st.session_state.chunks = {}  # Store text chunks by file_id
if "chunk_metadata" not in st.session_state:
    st.session_state.chunk_metadata = {}  # Store metadata about chunks

# Utility Functions
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = DocxDocument(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def extract_text_from_image(file):
    try:
        img = Image.open(file)
        return pytesseract.image_to_string(img)
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into overlapping chunks of approximately chunk_size characters"""
    if not text:
        return []

    # First try to split by sentences to preserve context
    try:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed chunk_size, save current chunk and start a new one
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Keep some overlap for context
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk

            current_chunk += " " + sentence

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks
    except Exception as e:
        # Fallback to simple text wrapping if sentence tokenization fails
        st.warning(f"Sentence tokenization failed: {str(e)}. Using fallback chunking method.")
        return textwrap.wrap(text, chunk_size, break_long_words=False, replace_whitespace=False)

def calculate_document_hash(content, metadata=None):
    """
    Calculate a hash for a document based on its content and metadata

    Args:
        content: The text content of the document
        metadata: Optional dictionary of metadata about the document

    Returns:
        A hash string that uniquely identifies the document
    """
    # Create a string that combines content and metadata
    hash_input = content

    # Add metadata if provided
    if metadata:
        for key, value in sorted(metadata.items()):
            hash_input += f"|{key}:{value}"

    # Calculate MD5 hash
    return hashlib.md5(hash_input.encode()).hexdigest()

def is_duplicate_document(content, metadata=None):
    """
    Check if a document with the same content and metadata has already been processed

    Args:
        content: The text content of the document
        metadata: Optional dictionary of metadata about the document

    Returns:
        (bool, str): Tuple of (is_duplicate, existing_file_id)
    """
    doc_hash = calculate_document_hash(content, metadata)

    # Check if this hash exists in our file_hashes
    if doc_hash in st.session_state.file_hashes:
        return True, st.session_state.file_hashes[doc_hash]

    return False, None

def generate_chunk_id(text, file_id, chunk_index):
    """Generate a unique ID for a chunk based on its content and metadata"""
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{file_id}_{chunk_index}_{content_hash}"

def extract_entities_with_openai(text, chunk_id=None):
    """
    Extract entities and relationships from text using OpenAI

    Args:
        text: The text to extract entities from
        chunk_id: Optional ID of the chunk (used for logging/debugging)
    """
    if not chat:
        return "OpenAI connection failed. Please check your API key."

    # Log which chunk we're processing if chunk_id is provided
    if chunk_id and st.session_state.get('debug_mode', False):
        st.write(f"Processing chunk: {chunk_id}")

    prompt = f"""
    Extract key entities and relationships from the following content as triples (subject - relation - object).
    Focus on identifying important entities and their relationships.
    Be specific and detailed in the relationships.
    Return ONLY a list of triples, one per line, in the format: subject - relation - object

    Content:
    {text[:4000]}  # Limit text to avoid token limits
    """
    try:
        response = chat([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        st.error(f"Error extracting entities: {str(e)}")
        return ""

def parse_triples(triples_text):
    """Parse triples from various formats that might be returned by the LLM"""
    parsed_triples = []

    # Try to parse as list items (1. subject - relation - object)
    numbered_pattern = r'\d+\.\s*(.*?)\s*-\s*(.*?)\s*-\s*(.*?)(?:\n|$)'
    matches = re.findall(numbered_pattern, triples_text)
    if matches:
        return [(s.strip(), r.strip(), o.strip()) for s, r, o in matches]

    # Try to parse as plain triples (subject - relation - object)
    for line in triples_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if ' - ' in line:
            parts = line.split(' - ')
            if len(parts) >= 3:
                subject = parts[0].strip()
                relation = parts[1].strip()
                obj = ' - '.join(parts[2:]).strip()  # Join in case object contains hyphens
                parsed_triples.append((subject, relation, obj))

    return parsed_triples

def populate_neo4j(triples_text, file_id, chunk_id=None, chunk_text=None):
    """
    Populate Neo4j with triples extracted from text.
    If chunk_id is provided, associate the triples with that chunk.
    """
    if not driver:
        st.error("Neo4j connection failed. Please check your connection settings.")
        return False

    triples = parse_triples(triples_text)
    if not triples:
        st.warning("No valid triples found in the extracted text.")
        return False

    try:
        with driver.session() as session:
            # If this is a chunk, create or update the Chunk node
            if chunk_id and chunk_text:
                # Create a Chunk node
                session.run(
                    """
                    MERGE (c:Chunk {id: $chunk_id, file_id: $file_id})
                    SET c.text = $text
                    """,
                    chunk_id=chunk_id, file_id=file_id, text=chunk_text[:1000]  # Store a preview of the text
                )

            # Process each triple
            for subject, relation, obj in triples:
                # Create the basic triple structure
                query = """
                MERGE (s:Entity {name: $subject, file_id: $file_id})
                MERGE (o:Entity {name: $object, file_id: $file_id})
                MERGE (s)-[r:RELATION {type: $relation, file_id: $file_id}]->(o)
                """

                # If this is from a chunk, connect entities to the chunk
                if chunk_id:
                    query += """
                    WITH s, o
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (s)-[:APPEARS_IN]->(c)
                    MERGE (o)-[:APPEARS_IN]->(c)
                    """

                session.run(
                    query,
                    subject=subject, object=obj, relation=relation,
                    file_id=file_id, chunk_id=chunk_id
                )
        return True
    except Exception as e:
        st.error(f"Error populating Neo4j: {str(e)}")
        return False

def display_graph(file_id=None, keywords=None, title="Knowledge Graph", height=500):
    """
    Display a knowledge graph visualization

    Args:
        file_id: ID of the file to filter by
        keywords: Optional list of keywords to filter entities
        title: Title for the graph section
        height: Height of the graph in pixels
    """
    # Display title if provided (but not if it's already displayed as a subheader)
    if title and not any(title in s for s in ["Knowledge-Query-Graph", "Knowledge Graph"]):
        st.subheader(title)

    if not driver:
        st.error("Neo4j connection failed. Please check your connection settings.")
        return

    G = nx.DiGraph()
    try:
        with driver.session() as session:
            # Determine which file_id to use
            current_id = file_id if file_id else st.session_state.current_file_id

            # Base query - will be modified based on parameters
            if keywords:
                # Query for specific keywords
                query = """
                MATCH (a:Entity {file_id: $file_id})
                WHERE
                """
                # Add conditions for each keyword
                keyword_conditions = []
                for keyword in keywords:
                    keyword_conditions.append(f"toLower(a.name) CONTAINS toLower('{keyword}')")

                if not keyword_conditions:
                    keyword_conditions.append("TRUE")  # Fallback if no keywords

                query += " OR ".join(keyword_conditions)
                query += """
                MATCH (a)-[r:RELATION]->(b:Entity {file_id: $file_id})
                RETURN a.name AS from, r.type AS rel, b.name AS to
                UNION
                MATCH (a:Entity {file_id: $file_id})-[r:RELATION]->(b:Entity {file_id: $file_id})
                WHERE
                """
                b_conditions = []
                for keyword in keywords:
                    b_conditions.append(f"toLower(b.name) CONTAINS toLower('{keyword}')")
                query += " OR ".join(b_conditions)
                query += """
                RETURN a.name AS from, r.type AS rel, b.name AS to
                """
            else:
                # Standard query for all relationships in the document
                query = """
                MATCH (a:Entity {file_id: $file_id})-[r:RELATION]->(b:Entity {file_id: $file_id})
                RETURN a.name AS from, r.type AS rel, b.name AS to
                """

            # Execute the query
            result = session.run(query, file_id=current_id)

            # Build the graph
            for record in result:
                G.add_node(record["from"])
                G.add_node(record["to"])
                G.add_edge(record["from"], record["to"], label=record["rel"])
    except Exception as e:
        st.error(f"Error querying Neo4j: {str(e)}")
        return

    if not G.nodes():
        st.info(f"No relationships found for this {'query' if keywords else 'document'}.")
        return

    try:
        net = Network(height=f"{height}px", width="100%", directed=True, notebook=False)
        net.from_nx(G)

        # Add physics options for better visualization
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 100
            }
          }
        }
        """)

        # Save and display the graph
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, "r", encoding="utf-8") as f:
            html_content = f.read()

        components.html(html_content, height=height+50)

        # Download button
        current_id = file_id if file_id else st.session_state.current_file_id
        graph_type = "query" if keywords else "full"
        st.download_button(
            label="Download Graph as HTML",
            data=html_content,
            file_name=f"{graph_type}_graph_{current_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.html",
            mime="text/html"
        )
    except Exception as e:
        st.error(f"Error creating graph visualization: {str(e)}")

def safe_cypher_query(cypher):
    """Validate and sanitize Cypher query to prevent injection and common errors"""
    # Basic validation - this is not comprehensive
    if not cypher.strip().upper().startswith("MATCH"):
        return None, "Query must start with MATCH"

    # Check for pattern expressions in RETURN clause which can cause errors
    if re.search(r'RETURN\s+.*?\(.*?\)-\[.*?\]->.*?\)', cypher, re.IGNORECASE):
        # Fix by modifying the query to use proper variable binding
        return None, "Pattern expressions in RETURN clause are not allowed"

    # Add file_id filter if not present
    if "file_id" not in cypher and st.session_state.current_file_id:
        # This is a simplistic approach - a more robust parser would be better
        cypher = cypher.replace("MATCH (", f"MATCH (n WHERE n.file_id = '{st.session_state.current_file_id}') WITH n MATCH (")

    return cypher, None

def extract_keywords_from_question(question):
    """Extract key entities and concepts from the question"""
    if not chat:
        return []

    prompt = f"""
    Extract the key entities, concepts, and search terms from this question.
    Return ONLY a comma-separated list of important terms (nouns, proper nouns, key concepts).

    Question: {question}
    """

    try:
        response = chat([HumanMessage(content=prompt)])
        keywords = [kw.strip() for kw in response.content.split(',')]
        return [kw for kw in keywords if kw]  # Filter out empty strings
    except Exception as e:
        st.error(f"Error extracting keywords: {str(e)}")
        # Fallback to simple word extraction
        return [w for w in re.findall(r'\b\w+\b', question) if len(w) > 3]

def query_neo4j_for_question(question):
    """Generate and execute Cypher queries based on the question"""
    if not chat:
        return "MATCH (n) RETURN n LIMIT 5", "OpenAI connection failed"

    # Extract keywords from the question
    keywords = extract_keywords_from_question(question)

    cypher_prompt = f"""
    Given the user question: "{question}", write a Cypher query that could help answer it using a graph of entities and relationships.

    The graph schema is:
    - Nodes have label 'Entity' with properties 'name' and 'file_id'
    - Relationships have type 'RELATION' with property 'type' containing the relation name
    - Nodes are connected to Chunk nodes via APPEARS_IN relationships
    - Chunk nodes have 'id', 'file_id', and 'text' properties

    Important keywords from the question: {', '.join(keywords)}

    Your query should:
    1. Find entities that match or are related to these keywords
    2. Explore relationships between these entities
    3. Return the relevant entities, relationships, and the text chunks they appear in
    4. Filter by file_id = '{st.session_state.current_file_id}'

    Return ONLY the Cypher query without any explanation or markdown formatting.
    """

    try:
        cypher_response = chat([HumanMessage(content=cypher_prompt)]).content
        # Extract the query if it's wrapped in backticks
        if "```" in cypher_response:
            cypher_response = re.search(r'```(?:cypher)?(.*?)```', cypher_response, re.DOTALL).group(1).strip()

        cypher, error = safe_cypher_query(cypher_response)
        if error:
            # Fallback query that searches for entities matching keywords
            fallback_query = f"""
            MATCH (e:Entity)
            WHERE e.file_id = '{st.session_state.current_file_id}'
            AND (
            """

            # Add conditions for each keyword
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append(f"toLower(e.name) CONTAINS toLower('{keyword}')")

            if not keyword_conditions:
                keyword_conditions.append("TRUE")  # Fallback if no keywords

            fallback_query += " OR ".join(keyword_conditions)
            fallback_query += """
            )
            OPTIONAL MATCH (e)-[r:RELATION]-(related:Entity)
            WHERE related.file_id = e.file_id
            OPTIONAL MATCH (e)-[:APPEARS_IN]->(c:Chunk)
            RETURN e.name AS entity,
                   type(r) AS relation,
                   related.name AS related_entity,
                   c.text AS context
            LIMIT 20
            """

            return fallback_query, error

        return cypher, None
    except Exception as e:
        # Return a simple fallback query
        return f"""
        MATCH (e:Entity)
        WHERE e.file_id = '{st.session_state.current_file_id}'
        OPTIONAL MATCH (e)-[:APPEARS_IN]->(c:Chunk)
        RETURN e.name AS entity, c.text AS context
        LIMIT 10
        """, str(e)

def run_hybrid_rag(question, file_id):
    """
    Run a hybrid RAG process that:
    1. Extracts keywords from the question
    2. Queries the knowledge graph for relevant entities and relationships
    3. Retrieves the text chunks containing those entities
    4. Generates an answer using the graph data and text chunks
    """
    if not driver or not chat:
        return "Error: Neo4j or OpenAI connection failed. Please check your settings.", ""

    # Set the current file_id to ensure queries target the right document
    st.session_state.current_file_id = file_id

    # Extract keywords for better context
    keywords = extract_keywords_from_question(question)
    st.session_state.last_keywords = keywords  # Store for display

    # Generate and run Cypher query
    cypher, error = query_neo4j_for_question(question)
    if error:
        st.warning(f"Query generation warning: {error}. Using fallback query.")

    # Execute the query
    graph_data = []
    chunk_texts = []
    entity_names = set()
    try:
        with driver.session() as session:
            try:
                # Try to run the generated query
                results = session.run(cypher)
                records = list(results)
            except Exception as query_error:
                # If the query fails, log the error and use a fallback query
                st.error(f"Generated query failed: {str(query_error)}")
                st.info("Using fallback query instead...")

                # Use a simple, reliable fallback query
                fallback_query = f"""
                MATCH (e:Entity)
                WHERE e.file_id = '{file_id}'
                RETURN e.name as entity
                LIMIT 50
                """
                results = session.run(fallback_query)
                records = list(results)

            if not records:
                st.info("No direct matches found. Trying broader search...")
                # Try a broader search if no results
                broader_query = f"""
                MATCH (e:Entity)
                WHERE e.file_id = '{file_id}'
                AND ANY(keyword IN {keywords} WHERE toLower(e.name) CONTAINS toLower(keyword))
                OPTIONAL MATCH (e)-[r:RELATION]-(related:Entity)
                WHERE related.file_id = e.file_id
                OPTIONAL MATCH (e)-[:APPEARS_IN]->(c:Chunk)
                RETURN e.name AS entity, r.type AS relation, related.name AS related_entity,
                       c.id AS chunk_id, c.text AS chunk_text
                LIMIT 15
                """
                try:
                    results = session.run(broader_query)
                    records = list(results)
                except Exception as broader_error:
                    st.error(f"Broader search query failed: {str(broader_error)}")
                    # Use an even simpler query as last resort
                    simplest_query = f"""
                    MATCH (e:Entity)
                    WHERE e.file_id = '{file_id}'
                    RETURN e.name as entity
                    LIMIT 10
                    """
                    results = session.run(simplest_query)
                    records = list(results)

            # Process the results
            for record in records:
                data = record.data()
                graph_data.append(data)

                # Collect entity names for context - handle different result formats
                for key, value in data.items():
                    # Add any field that might contain an entity name
                    if key in ['entity', 'from', 'to', 'related_entity'] and value:
                        entity_names.add(value)
                    # Add any field that might contain chunk text
                    if key in ['chunk_text', 'context', 'text'] and value and value not in chunk_texts:
                        chunk_texts.append(value)

            # If no chunks were found directly, try to find chunks containing the entities
            if not chunk_texts and entity_names:
                try:
                    chunk_query = f"""
                    MATCH (e:Entity)-[:APPEARS_IN]->(c:Chunk)
                    WHERE e.file_id = '{file_id}' AND e.name IN {list(entity_names)}
                    RETURN DISTINCT c.text AS chunk_text
                    LIMIT 5
                    """
                    chunk_results = session.run(chunk_query)
                    for chunk_record in chunk_results:
                        if 'chunk_text' in chunk_record and chunk_record['chunk_text'] and chunk_record['chunk_text'] not in chunk_texts:
                            chunk_texts.append(chunk_record['chunk_text'])
                except Exception as chunk_error:
                    st.warning(f"Could not retrieve chunks: {str(chunk_error)}")

                    # Try a simpler query as fallback
                    try:
                        simple_chunk_query = f"""
                        MATCH (c:Chunk)
                        WHERE c.file_id = '{file_id}'
                        RETURN c.text AS chunk_text
                        LIMIT 3
                        """
                        simple_results = session.run(simple_chunk_query)
                        for chunk_record in simple_results:
                            if 'chunk_text' in chunk_record and chunk_record['chunk_text'] and chunk_record['chunk_text'] not in chunk_texts:
                                chunk_texts.append(chunk_record['chunk_text'])
                    except Exception:
                        # If all else fails, just continue without chunks
                        pass
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return f"Error querying the knowledge graph: {str(e)}", cypher

    # Prepare data for the answer generation
    if not graph_data:
        extracted_info = "No relevant information found in the knowledge graph."
    else:
        # Format the graph data for better readability
        formatted_data = []

        for item in graph_data:
            # Handle different result formats
            entry = ""

            # Case 1: Standard entity-relation-entity format
            if 'entity' in item and item['entity']:
                entry = f"Entity: {item['entity']}"
                if 'relation' in item and item['relation'] and 'related_entity' in item and item['related_entity']:
                    entry += f" | {item['relation']} | {item['related_entity']}"

            # Case 2: Graph format with from-rel-to
            elif 'from' in item and item['from']:
                entry = f"Entity: {item['from']}"
                if 'rel' in item and item['rel'] and 'to' in item and item['to']:
                    entry += f" | {item['rel']} | {item['to']}"

            # Add any other properties that might be useful
            extra_props = []
            for key, value in item.items():
                if key not in ['entity', 'relation', 'related_entity', 'from', 'rel', 'to', 'chunk_text', 'chunk_id', 'text', 'context'] and value:
                    extra_props.append(f"{key}: {value}")

            if extra_props:
                entry += f" ({', '.join(extra_props)})"

            if entry:
                formatted_data.append(entry)

        extracted_info = "\n".join(formatted_data)

    # Combine graph data with chunk texts
    context = f"""
    Graph Data:
    {extracted_info}

    Relevant Text Chunks:
    {' '.join(chunk_texts[:3])}  # Limit to first 3 chunks to avoid token limits
    """

    # Generate the answer
    hybrid_prompt = f"""
    Answer the following question based on the provided context: "{question}"

    Context:
    {context}

    Important entities: {', '.join(entity_names)}

    Instructions:
    1. Focus on the specific entities mentioned in the question: {', '.join(keywords)}
    2. Use information from both the graph relationships and the text chunks
    3. If the context doesn't contain enough information to answer completely, state that clearly
    4. Provide a concise but comprehensive answer
    """

    try:
        answer = chat([HumanMessage(content=hybrid_prompt)])
        return answer.content, cypher
    except Exception as e:
        return f"Error generating answer: {str(e)}", cypher

# File upload and processing
col1, col2 = st.columns([2, 5])

with col1:
    st.subheader("Upload Documents")

    # Chunking settings
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size (characters)", 500, 5000, 2000, 500)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 500, 200, 50)

        # Initialize debug_mode in session state if it doesn't exist
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = False

        # Debug mode toggle
        debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        st.session_state.debug_mode = debug_mode

        if debug_mode:
            st.info("Debug mode enabled. Additional information will be shown during processing.")

    uploaded_files = st.file_uploader("Upload PDFs, Word Docs, or Images",
                                     type=["pdf", "docx", "png", "jpg", "jpeg"],
                                     accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                file_type = file.name.split(".")[-1].lower()

                st.info(f"Processing {file.name}...")
                text = ""

                # Extract text based on file type
                if file_type == "pdf":
                    text = extract_text_from_pdf(file)
                elif file_type == "docx":
                    text = extract_text_from_docx(file)
                elif file_type in ["png", "jpg", "jpeg"]:
                    text = extract_text_from_image(file)

                if not text:
                    st.error(f"Could not extract text from {file.name}")
                    continue

                # Collect metadata about the document
                try:
                    file_size = len(file.getvalue())
                    file_stats = {
                        "name": file.name,
                        "type": file_type,
                        "size": file_size,
                        "content_length": len(text),
                        "content_preview": text[:100]
                    }

                    # For PDFs, try to extract more metadata
                    if file_type == "pdf":
                        reader = PdfReader(file)
                        file_stats["pages"] = len(reader.pages)
                        if reader.metadata:
                            for key in reader.metadata:
                                if key and reader.metadata[key]:
                                    file_stats[f"pdf_{key}"] = str(reader.metadata[key])
                except Exception as e:
                    st.warning(f"Could not extract complete metadata: {str(e)}")

                # Check if this document is a duplicate
                is_duplicate, existing_id = is_duplicate_document(text, file_stats)

                if is_duplicate:
                    st.warning(f"This document appears to be identical to a previously processed document.")

                    # Ask user if they want to use the existing document
                    if st.button(f"Use existing document '{st.session_state.file_names.get(existing_id, 'Unknown')}'"):
                        st.session_state.current_file_id = existing_id
                        st.success(f"Switched to existing document.")
                        st.experimental_rerun()
                    continue

                # Generate a new file ID
                file_id = str(uuid4())[:8]

                # Calculate document hash and store it
                doc_hash = calculate_document_hash(text, file_stats)
                st.session_state.file_hashes[doc_hash] = file_id
                st.session_state.file_metadata[file_id] = file_stats

                # Chunk the text
                st.info(f"Chunking text from {file.name}...")
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)

                if not chunks:
                    st.error(f"Failed to chunk text from {file.name}")
                    continue

                # Store chunks in session state
                st.session_state.chunks[file_id] = chunks

                # Process each chunk
                chunk_progress = st.progress(0)
                chunk_status = st.empty()

                for i, chunk in enumerate(chunks):
                    progress_pct = (i + 1) / len(chunks)
                    chunk_progress.progress(progress_pct)
                    chunk_status.info(f"Processing chunk {i+1}/{len(chunks)}")

                    # Generate a unique ID for this chunk
                    chunk_id = generate_chunk_id(chunk, file_id, i)

                    # Store metadata about this chunk
                    if file_id not in st.session_state.chunk_metadata:
                        st.session_state.chunk_metadata[file_id] = {}

                    st.session_state.chunk_metadata[file_id][chunk_id] = {
                        "index": i,
                        "start_char": i * (chunk_size - chunk_overlap) if i > 0 else 0,
                        "text_preview": chunk[:100] + "..."
                    }

                    # Extract entities from this chunk
                    triples = extract_entities_with_openai(chunk, chunk_id)

                    # Populate Neo4j with the triples from this chunk
                    populate_neo4j(triples, file_id, chunk_id, chunk)

                # Clear progress indicators
                chunk_progress.empty()
                chunk_status.success(f"Processed {len(chunks)} chunks from {file.name}")

                # Store file ID and name in session state
                if file_id not in st.session_state.file_ids:
                    st.session_state.file_ids.append(file_id)
                    st.session_state.file_names[file_id] = file.name

                # Set as current file if it's the first one
                if not st.session_state.current_file_id:
                    st.session_state.current_file_id = file_id

                st.success(f"Successfully processed {file.name} into {len(chunks)} chunks")

# Document selector and question input
with col2:
    st.subheader("Ask Questions")

    # Document selector
    if st.session_state.file_ids:
        file_options = {st.session_state.file_names[fid]: fid for fid in st.session_state.file_ids}
        selected_file = st.selectbox(
            "Select document to query:",
            options=list(file_options.keys()),
            index=list(file_options.keys()).index(
                st.session_state.file_names[st.session_state.current_file_id]
            ) if st.session_state.current_file_id in st.session_state.file_names else 0
        )
        st.session_state.current_file_id = file_options[selected_file]

        # Show document metadata
        with st.expander("Document Information"):
            current_id = st.session_state.current_file_id

            # Show basic info
            st.write(f"**Document ID:** {current_id}")
            st.write(f"**Name:** {st.session_state.file_names.get(current_id, 'Unknown')}")

            # Show metadata if available
            if current_id in st.session_state.file_metadata:
                metadata = st.session_state.file_metadata[current_id]

                # Create two columns for metadata display
                meta_col1, meta_col2 = st.columns(2)

                with meta_col1:
                    st.write("**Basic Information:**")
                    if "type" in metadata:
                        st.write(f"Type: {metadata['type'].upper()}")
                    if "size" in metadata:
                        st.write(f"Size: {metadata['size'] / 1024:.1f} KB")
                    if "content_length" in metadata:
                        st.write(f"Content Length: {metadata['content_length']} characters")
                    if "pages" in metadata:
                        st.write(f"Pages: {metadata['pages']}")

                with meta_col2:
                    st.write("**Additional Information:**")
                    # Show PDF metadata if available
                    pdf_meta = {k: v for k, v in metadata.items() if k.startswith("pdf_")}
                    if pdf_meta:
                        for key, value in pdf_meta.items():
                            display_key = key.replace("pdf_", "").title()
                            st.write(f"{display_key}: {value}")

            # Show chunk information
            if current_id in st.session_state.chunks:
                num_chunks = len(st.session_state.chunks[current_id])
                st.write(f"**Chunks:** {num_chunks}")

            # Show hash information if in debug mode
            if st.session_state.get('debug_mode', False):
                # Find the hash for this file_id
                for doc_hash, fid in st.session_state.file_hashes.items():
                    if fid == current_id:
                        st.write(f"**Document Hash:** {doc_hash}")
                        break

    # Question input
    question = st.text_input("Ask a question about your documents:")

    if st.button("Submit Question") and question and st.session_state.current_file_id:
        with st.spinner("Generating answer..."):
            answer, cypher_query = run_hybrid_rag(question, st.session_state.current_file_id)
            st.session_state.history.append((question, answer, st.session_state.current_file_id))

            st.subheader("Answer")
            st.write(answer)

            # Show extracted keywords if available
            if hasattr(st.session_state, 'last_keywords') and st.session_state.last_keywords:
                with st.expander("Keywords Extracted from Question"):
                    st.write(", ".join(st.session_state.last_keywords))

            # Show the Cypher query
            with st.expander("Generated Cypher Query"):
                st.code(cypher_query, language="cypher")

            # Show document chunks info
            if st.session_state.current_file_id in st.session_state.chunks:
                num_chunks = len(st.session_state.chunks[st.session_state.current_file_id])
                with st.expander(f"Document Chunks ({num_chunks})"):
                    st.write(f"This document was processed into {num_chunks} chunks for better analysis.")

                    # Show a sample of chunks
                    if st.session_state.current_file_id in st.session_state.chunk_metadata:
                        metadata = st.session_state.chunk_metadata[st.session_state.current_file_id]
                        for chunk_id, info in list(metadata.items())[:5]:  # Show first 5 chunks
                            st.markdown(f"**Chunk {info['index']+1}**: {info['text_preview']}")

                    st.info("Chunking helps the system analyze large documents by breaking them into manageable pieces while maintaining context.")

# Query-specific Knowledge Graph (only shown after a question is asked)
if hasattr(st.session_state, 'last_keywords') and st.session_state.last_keywords and st.session_state.current_file_id:
    st.subheader("Knowledge-Query-Graph")
    st.caption("Showing relationships related to your question")
    display_graph(
        file_id=st.session_state.current_file_id,
        keywords=st.session_state.last_keywords,
        title="Knowledge-Query-Graph",
        height=400
    )

# Full Knowledge Graph
st.subheader("Knowledge Graph")
st.caption("Showing all relationships in the document")
if st.session_state.current_file_id:
    display_graph(st.session_state.current_file_id, title="Knowledge Graph")
else:
    st.info("Upload and select a document to view its knowledge graph.")

# Show chat history in sidebar
for i, (q, a, fid) in enumerate(reversed(st.session_state.history)):
    with st.sidebar.expander(f"Q{i+1}: {q[:30]}..." if len(q) > 30 else f"Q{i+1}: {q}"):
        st.write(f"**Document:** {st.session_state.file_names.get(fid, 'Unknown')}")
        st.write(f"**Answer:** {a}")
        if st.button(f"Show graph for this question", key=f"history_{i}"):
            st.session_state.current_file_id = fid
            st.experimental_rerun()
