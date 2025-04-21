import streamlit as st
import os
import tempfile
import uuid
import json
import datetime
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.documents import Document
from neo4j.exceptions import ClientError
import base64

# Try to import image processing libraries, but make them optional
IMAGE_SUPPORT = False
try:
    from PIL import Image
    IMAGE_SUPPORT = True
except ImportError:
    pass

# Define embedding model options
EMBEDDING_MODELS = {
    "text-embedding-3-small": {
        "name": "OpenAI text-embedding-3-small",
        "dimensions": 1536,
        "supports_multimodal": False,
        "description": "Efficient text embedding model with 1536 dimensions"
    },
    "text-embedding-3-large": {
        "name": "OpenAI text-embedding-3-large",
        "dimensions": 3072,
        "supports_multimodal": False,
        "description": "High-performance text embedding model with 3072 dimensions"
    },
    "text-embedding-ada-002": {
        "name": "OpenAI text-embedding-ada-002 (Legacy)",
        "dimensions": 1536,
        "supports_multimodal": False,
        "description": "Legacy text embedding model with 1536 dimensions"
    }
}

# Custom multimodal embedding class for handling images
class MultimodalEmbeddings(OpenAIEmbeddings):
    """Custom embedding class that handles both text and images"""

    def embed_documents(self, texts, images=None):
        """Generate embeddings for a list of documents, with optional image data"""
        # Process each document to check for image data in metadata
        processed_texts = []

        for text in texts:
            # Check if this is a Document object with image metadata
            if hasattr(text, 'metadata') and text.metadata.get('is_image') and text.metadata.get('image_base64'):
                # This is an image document, use the image data for embedding
                processed_texts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{text.metadata.get('format', 'jpeg').lower()};base64,{text.metadata['image_base64']}"
                    }
                })
            else:
                # Regular text document
                if hasattr(text, 'page_content'):
                    processed_texts.append(text.page_content)
                else:
                    processed_texts.append(text)

        # Call the parent class's embed_documents method with the processed texts
        return super().embed_documents(processed_texts)

# Function to get image base64 encoding for multimodal embeddings
def get_image_base64(image_path):
    """Convert an image to base64 encoding for use with multimodal embeddings"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to check and fix Neo4j database before creating a new index
def check_and_fix_neo4j_database(url, username, password):
    """Check Neo4j database for existing vector index and clean up if needed"""
    try:
        from neo4j import GraphDatabase

        # Connect to Neo4j
        driver = GraphDatabase.driver(url, auth=(username, password))

        # Check if vector index exists and get its dimension
        with driver.session() as session:
            # Check if the vector index exists
            index_result = session.run("""
                SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options
                WHERE name = 'vector'
                RETURN name, options
            """)

            index_data = index_result.single()
            if index_data:
                show_status("Found existing vector index in Neo4j", "info")
                # Get the dimension of the existing index if available
                try:
                    index_options = index_data["options"]
                    if "indexConfig" in index_options and "vector.dimensions" in index_options["indexConfig"]:
                        existing_dimension = index_options["indexConfig"]["vector.dimensions"]
                        show_status(f"Existing vector index has dimension: {existing_dimension}", "info")
                        return {"exists": True, "dimension": existing_dimension}
                except Exception as e:
                    show_status(f"Could not determine existing index dimension: {e}", "warning")

            return {"exists": False}
    except Exception as e:
        show_status(f"Error checking Neo4j database: {e}", "error")
        return {"exists": False, "error": str(e)}
    finally:
        if 'driver' in locals():
            driver.close()

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
neo4j_url = os.getenv("NEO4J_URL")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for filters
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'file_type': [],
        'file_name': [],
        'upload_date': []
    }

# Function to load chat history from disk
def load_chat_history():
    try:
        if os.path.exists('chat_history.json'):
            with open('chat_history.json', 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return []

# Function to save chat history to disk
def save_chat_history(history):
    try:
        with open('chat_history.json', 'w') as f:
            json.dump(history, f)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

# Load chat history from disk if not already in session state
if len(st.session_state.chat_history) == 0:
    st.session_state.chat_history = load_chat_history()

# App UI
st.set_page_config(page_title="KnowledgeWeaver", layout="wide")

# Custom CSS for better spacing and readability
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        color: #842029;
    }
    .info-box {
        background-color: #cff4fc;
        color: #055160;
    }
</style>
""", unsafe_allow_html=True)

# Custom status message functions
def show_status(message, status_type="info"):
    """Display a styled status message

    Args:
        message: The message to display
        status_type: One of 'success', 'info', 'warning', or 'error'
    """
    if status_type == "success":
        st.markdown(f"<div class='status-box success-box'>✅ {message}</div>", unsafe_allow_html=True)
    elif status_type == "warning":
        st.markdown(f"<div class='status-box warning-box'>⚠️ {message}</div>", unsafe_allow_html=True)
    elif status_type == "error":
        st.markdown(f"<div class='status-box error-box'>❌ {message}</div>", unsafe_allow_html=True)
    else:  # info
        st.markdown(f"<div class='status-box info-box'>ℹ️ {message}</div>", unsafe_allow_html=True)

st.title("📄🔍 KnowledgeWeaver-V1")
st.markdown("### Advanced Document Management with Multi-format Support and Filtering")

# Add navigation links
st.markdown("""
<div style='display: flex; gap: 1rem; margin-bottom: 1rem;'>
    <a href='/' style='text-decoration: none;'>
        <div style='background-color: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; text-align: center;'>
            📁 Document Management
        </div>
    </a>
    <a href='/image_analysis' style='text-decoration: none;'>
        <div style='background-color: #2196F3; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; text-align: center;'>
            📷 Image Analysis
        </div>
    </a>
</div>
""", unsafe_allow_html=True)

# Create sidebar for filters and chat history
sidebar = st.sidebar
sidebar.title("📋 Options & History")

# Chat History Section in Sidebar
sidebar.header("💬 Chat History")
if st.session_state.chat_history:
    for i, entry in enumerate(st.session_state.chat_history[-10:]):
        with sidebar.expander(f"Q: {entry['question'][:30]}..."):
            st.write(f"**Question:** {entry['question']}")
            st.write(f"**Answer:** {entry['answer'][:100]}...")
            st.write(f"**Time:** {entry['timestamp']}")

# Filter Section in Sidebar
sidebar.header("🔍 Filters")

# Document Type Filter
if 'all_file_types' in st.session_state:
    selected_file_types = sidebar.multiselect(
        "Filter by Document Type",
        options=st.session_state.all_file_types,
        default=[]
    )
    if selected_file_types:
        st.session_state.filters['file_type'] = selected_file_types

# Document Name Filter
if 'all_file_names' in st.session_state:
    selected_file_names = sidebar.multiselect(
        "Filter by Document Name",
        options=st.session_state.all_file_names,
        default=[]
    )
    if selected_file_names:
        st.session_state.filters['file_name'] = selected_file_names

# Upload Date Filter
if 'all_upload_dates' in st.session_state:
    selected_dates = sidebar.multiselect(
        "Filter by Upload Date",
        options=st.session_state.all_upload_dates,
        default=[]
    )
    if selected_dates:
        st.session_state.filters['upload_date'] = selected_dates

# Main content area
main_content = st.container()
with main_content:
    # Create two columns for file upload and embedding model selection
    upload_col, model_col = st.columns([3, 2])

    with upload_col:
        uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "jpg", "png"], accept_multiple_files=True)

    with model_col:
        # Embedding model selection
        embedding_model = st.selectbox(
            "Select Embedding Model",
            options=list(EMBEDDING_MODELS.keys()),
            format_func=lambda x: EMBEDDING_MODELS[x]["name"],
            index=0,  # Default to text-embedding-3-small
            help="Select the embedding model to use for document vectorization"
        )

        # Show model details
        selected_model = EMBEDDING_MODELS[embedding_model]
        st.markdown(f"**Dimensions:** {selected_model['dimensions']}")
        st.markdown(f"**Description:** {selected_model['description']}")

if uploaded_files:
    # Initialize lists to collect all documents
    all_documents = []
    all_file_ids = []
    all_file_types = []
    all_file_names = []
    all_upload_dates = []

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Show processing status
        processing_status = st.empty()
        processing_status.markdown(f"<div class='status-box info-box'>⏳ Processing {uploaded_file.name}...</div>", unsafe_allow_html=True)

        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Get file extension and determine file type
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            file_type = file_extension.replace('.', '')

            # Current timestamp for upload date
            upload_date = datetime.now().strftime("%Y-%m-%d")

            # Process different file types
            documents = []
            try:
                if file_extension == ".pdf":
                    show_status(f"Processing PDF file: {uploaded_file.name}", "info")
                    loader = PyMuPDFLoader(tmp_path)
                    documents = loader.load()
                    show_status(f"Extracted {len(documents)} pages from PDF", "success")
                elif file_extension == ".docx":
                    show_status(f"Processing Word document: {uploaded_file.name}", "info")
                    loader = UnstructuredWordDocumentLoader(tmp_path)
                    documents = loader.load()
                    show_status(f"Extracted content from Word document", "success")
                elif file_extension in [".jpg", ".png"] and IMAGE_SUPPORT:
                    # Process image files using PIL
                    show_status(f"Processing image file: {uploaded_file.name}", "info")
                    try:
                        img = Image.open(tmp_path)
                        # Extract basic image information
                        width, height = img.size
                        format_info = img.format
                        mode = img.mode

                        # Create a simple text description of the image
                        image_text = f"Image information:\nFormat: {format_info}\nSize: {width}x{height}\nMode: {mode}"

                        # Get base64 encoding of the image for multimodal embeddings
                        try:
                            image_base64 = get_image_base64(tmp_path)
                            # Store the base64 encoding in metadata for multimodal embeddings
                            image_metadata = {
                                "source": uploaded_file.name,
                                "image_base64": image_base64,
                                "is_image": True,
                                "width": width,
                                "height": height,
                                "format": format_info,
                                "mode": mode
                            }
                        except Exception as base64_error:
                            show_status(f"Warning: Could not create base64 encoding for image: {base64_error}", "warning")
                            image_metadata = {"source": uploaded_file.name, "is_image": True}

                        # Create a document with the image information
                        documents = [Document(page_content=image_text, metadata=image_metadata)]
                        show_status(f"Extracted metadata from image", "success")
                    except Exception as img_error:
                        show_status(f"Error processing image: {img_error}", "error")
                        st.exception(img_error)
                else:
                    if file_extension in [".jpg", ".png"] and not IMAGE_SUPPORT:
                        show_status("Image support is not available. PIL library is required for image processing.", "warning")
                    else:
                        show_status(f"Unsupported file type: {file_extension}", "error")

                if documents:
                    # Generate a unique file_id for the uploaded file
                    file_id = str(uuid.uuid4())

                    # Add to our collection lists
                    all_documents.extend(documents)
                    all_file_ids.append(file_id)
                    all_file_types.append(file_type)
                    all_file_names.append(uploaded_file.name)
                    all_upload_dates.append(upload_date)

                    show_status(f"Successfully processed {uploaded_file.name}", "success")
                else:
                    show_status(f"No content could be extracted from {uploaded_file.name}.", "warning")

            except Exception as e:
                show_status(f"Error processing file {uploaded_file.name}: {e}", "error")
                st.exception(e)
        except Exception as e:
            show_status(f"Error reading file {uploaded_file.name}: {e}", "error")
            st.exception(e)
        finally:
            # Clear the processing status
            processing_status.empty()

    if all_documents:
        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_documents)

        # Add metadata to each chunk based on its source document
        for i, doc in enumerate(docs):
            # Find which original document this chunk came from
            # This is a simplification - in a real app you might need more sophisticated tracking
            # Here we're assuming the order is preserved from the original documents to chunks
            doc_index = 0
            doc_count = 0
            for j, count in enumerate([len(d) for d in all_documents]):
                if i >= doc_count and i < doc_count + count:
                    doc_index = j
                    break
                doc_count += count

            # Add metadata
            doc.metadata["file_id"] = all_file_ids[doc_index] if doc_index < len(all_file_ids) else str(uuid.uuid4())
            doc.metadata["file_name"] = all_file_names[doc_index] if doc_index < len(all_file_names) else "unknown"
            doc.metadata["file_type"] = all_file_types[doc_index] if doc_index < len(all_file_types) else "unknown"
            doc.metadata["upload_date"] = all_upload_dates[doc_index] if doc_index < len(all_upload_dates) else datetime.now().strftime("%Y-%m-%d")

        # Create embeddings based on selected model
        show_status(f"Using embedding model: {EMBEDDING_MODELS[embedding_model]['name']}", "info")

        # Check if any documents are images
        has_images = any(doc.metadata.get('is_image', False) for doc in docs)

        # Initialize embeddings with the selected model
        if has_images:
            show_status("Detected image documents, using multimodal embedding support", "info")
            embeddings = MultimodalEmbeddings(
                model=embedding_model,
                openai_api_key=openai_api_key,
                dimensions=EMBEDDING_MODELS[embedding_model]["dimensions"]
            )
        else:
            embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=openai_api_key,
                dimensions=EMBEDDING_MODELS[embedding_model]["dimensions"]
            )

        # Create vector index in Neo4j with error handling for dimension mismatch
        vector_status = st.empty()
        vector_status.markdown(f"<div class='status-box info-box'>⏳ Creating vector embeddings and storing in Neo4j...</div>", unsafe_allow_html=True)

        # Check Neo4j database for existing vector index
        db_check = check_and_fix_neo4j_database(neo4j_url, neo4j_user, neo4j_password)

        # If index exists and dimensions don't match, we need to recreate it
        recreate_index = False
        if db_check.get("exists", False):
            existing_dimension = db_check.get("dimension")
            current_dimension = EMBEDDING_MODELS[embedding_model]["dimensions"]

            if existing_dimension and existing_dimension != current_dimension:
                show_status(f"Dimension mismatch detected: Existing index has {existing_dimension} dimensions, but current model uses {current_dimension} dimensions", "warning")
                recreate_index = True

                # Try to drop the existing index
                try:
                    from neo4j import GraphDatabase
                    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
                    with driver.session() as session:
                        # Drop the vector index if it exists
                        session.run("DROP INDEX vector IF EXISTS")
                        # Also clean up any existing Document nodes to avoid conflicts
                        session.run("MATCH (d:Document) DETACH DELETE d")
                    driver.close()
                    show_status("Successfully dropped existing index with mismatched dimensions", "success")
                except Exception as e:
                    show_status(f"Error dropping existing index: {e}. Will attempt to recreate anyway.", "warning")

        try:
            # Create embeddings with progress indicator
            show_status(f"Creating embeddings for {len(docs)} document chunks using {EMBEDDING_MODELS[embedding_model]['name']}", "info")

            try:
                # If we need to recreate the index, we need to handle it manually
                if recreate_index:
                    # First create the vector store without documents
                    vector_index = Neo4jVector(
                        embedding=embeddings,
                        url=neo4j_url,
                        username=neo4j_user,
                        password=neo4j_password,
                        index_name="vector",
                        node_label="Document",
                        # Set the text key to match what will be used for documents
                        text_node_property="text",
                        embedding_node_property="embedding"
                    )

                    # Then add documents one by one
                    for i, doc in enumerate(docs):
                        if i % 10 == 0:  # Update status every 10 documents
                            show_status(f"Adding document {i+1}/{len(docs)} to Neo4j...", "info")
                        vector_index.add_documents([doc])
                else:
                    # Standard approach when no recreation is needed
                    vector_index = Neo4jVector.from_documents(
                        documents=docs,
                        embedding=embeddings,
                        url=neo4j_url,
                        username=neo4j_user,
                        password=neo4j_password,
                        index_name="vector",
                        node_label="Document"
                    )

                # Save vector in session (but not a specific file_id since we have multiple)
                st.session_state.vector = vector_index

                # Update available filters
                if 'all_file_types' not in st.session_state:
                    st.session_state.all_file_types = []
                if 'all_file_names' not in st.session_state:
                    st.session_state.all_file_names = []
                if 'all_upload_dates' not in st.session_state:
                    st.session_state.all_upload_dates = []

                # Add all file types, names, and dates to session state
                for file_type in all_file_types:
                    if file_type not in st.session_state.all_file_types:
                        st.session_state.all_file_types.append(file_type)

                for file_name in all_file_names:
                    if file_name not in st.session_state.all_file_names:
                        st.session_state.all_file_names.append(file_name)

                for upload_date in all_upload_dates:
                    if upload_date not in st.session_state.all_upload_dates:
                        st.session_state.all_upload_dates.append(upload_date)

                vector_status.empty()
                show_status(f"{len(uploaded_files)} document(s) uploaded and processed successfully!", "success")

            except (ClientError, ValueError) as e:
                error_message = str(e)
                # Check for various dimension mismatch error patterns
                if ("dimension mismatch" in error_message.lower() or
                    "1536 vs 4096" in error_message.lower() or
                    "dimensions do not match" in error_message.lower() or
                    "index with name vector already exists" in error_message.lower()):

                    show_status("Detected dimension mismatch in vector index. Attempting to fix...", "warning")
                    show_status(f"Error details: {error_message}", "info")

                    try:
                        # Delete the existing vector index
                        show_status("Deleting existing vector index...", "info")
                        try:
                            Neo4jVector.delete_index(
                                url=neo4j_url,
                                username=neo4j_user,
                                password=neo4j_password,
                                index_name="vector"
                            )
                            show_status("Successfully deleted existing vector index", "success")
                        except Exception as delete_error:
                            show_status(f"Error deleting index: {delete_error}. Attempting to continue anyway...", "warning")

                            # Try to execute a direct Cypher query to drop the index
                            try:
                                from neo4j import GraphDatabase
                                driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
                                with driver.session() as session:
                                    # Drop the vector index if it exists
                                    session.run("DROP INDEX vector IF EXISTS")
                                    # Also clean up any existing Document nodes to avoid conflicts
                                    session.run("MATCH (d:Document) DETACH DELETE d")
                                driver.close()
                                show_status("Successfully cleaned up database using direct queries", "success")
                            except Exception as cypher_error:
                                show_status(f"Error executing direct Cypher cleanup: {cypher_error}", "error")
                                st.exception(cypher_error)

                        # Recreate the vector index
                        show_status("Recreating vector index with correct dimensions...", "info")

                        # First create the vector store without documents
                        vector_index = Neo4jVector(
                            embedding=embeddings,
                            url=neo4j_url,
                            username=neo4j_user,
                            password=neo4j_password,
                            index_name="vector",
                            node_label="Document",
                            # Set the text key to match what will be used for documents
                            text_node_property="text",
                            embedding_node_property="embedding"
                        )

                        # Then add documents one by one
                        for i, doc in enumerate(docs):
                            if i % 10 == 0:  # Update status every 10 documents
                                show_status(f"Adding document {i+1}/{len(docs)} to Neo4j...", "info")
                            vector_index.add_documents([doc])

                        # Save vector in session (but not a specific file_id since we have multiple)
                        st.session_state.vector = vector_index

                        # Update available filters
                        if 'all_file_types' not in st.session_state:
                            st.session_state.all_file_types = []
                        if 'all_file_names' not in st.session_state:
                            st.session_state.all_file_names = []
                        if 'all_upload_dates' not in st.session_state:
                            st.session_state.all_upload_dates = []

                        # Add all file types, names, and dates to session state
                        for file_type in all_file_types:
                            if file_type not in st.session_state.all_file_types:
                                st.session_state.all_file_types.append(file_type)

                        for file_name in all_file_names:
                            if file_name not in st.session_state.all_file_names:
                                st.session_state.all_file_names.append(file_name)

                        for upload_date in all_upload_dates:
                            if upload_date not in st.session_state.all_upload_dates:
                                st.session_state.all_upload_dates.append(upload_date)

                        vector_status.empty()
                        show_status(f"Vector index fixed and {len(uploaded_files)} document(s) uploaded successfully!", "success")

                    except Exception as inner_e:
                        vector_status.empty()
                        show_status(f"Failed to fix vector index: {inner_e}", "error")
                        st.exception(inner_e)
                else:
                    vector_status.empty()
                    show_status(f"Error creating vector index: {e}", "error")
                    st.exception(e)
            except Exception as e:
                vector_status.empty()
                show_status(f"Error creating vector index: {e}", "error")
                st.exception(e)
        except Exception as e:
            vector_status.empty()
            show_status(f"Unexpected error during vector processing: {e}", "error")
            st.exception(e)

# Initialize ChatOpenAI LLM
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")  # Use ChatOpenAI for chat-based models like GPT-4

# Custom Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based strictly on the content provided.

First, give direct answer for the users question.

Then, give a brief overview for the answer in 5-6 sentences.

Then, list key points in bullet form.

Finally, mention the source filenames or identifiers used.

<context>
<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)

# Function to build filter based on selected filters
def build_filter_dict():
    filter_dict = {}

    # We no longer add a default file_id filter since we're handling multiple files
    # Only apply filters that the user has explicitly selected

    # Add other filters if selected
    if st.session_state.filters['file_type']:
        filter_dict["file_type"] = st.session_state.filters['file_type']
    if st.session_state.filters['file_name']:
        filter_dict["file_name"] = st.session_state.filters['file_name']
    if st.session_state.filters['upload_date']:
        filter_dict["upload_date"] = st.session_state.filters['upload_date']

    return filter_dict

# Function to export answer as JSON
def export_answer(question, answer, context_docs):
    export_data = {
        "question": question,
        "answer": answer,
        "context": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in context_docs
        ],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Create a download button for the JSON
    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="📥 Download Answer as JSON",
        data=json_str,
        file_name=f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# If vectors exist, set up retriever with filtering
if "vector" in st.session_state:
    # Display system status
    if not st.session_state.get('system_ready_shown', False):
        show_status("System ready for queries! You can now ask questions about your documents.", "success")
        st.session_state.system_ready_shown = True

    # Create a card-like container for the query section
    query_container = st.container()
    with query_container:
        st.markdown("### 🔍 Ask Questions About Your Documents")

        # Show active filters if any
        active_filters = []
        if st.session_state.filters['file_type']:
            active_filters.append(f"File Type: {', '.join(st.session_state.filters['file_type'])}")
        if st.session_state.filters['file_name']:
            active_filters.append(f"File Name: {', '.join(st.session_state.filters['file_name'])}")
        if st.session_state.filters['upload_date']:
            active_filters.append(f"Upload Date: {', '.join(st.session_state.filters['upload_date'])}")

        if active_filters:
            filter_text = " | ".join(active_filters)
            st.markdown(f"<div style='padding: 0.5rem; background-color: #f0f2f6; border-radius: 0.5rem; margin-bottom: 1rem;'><strong>Active Filters:</strong> {filter_text}</div>", unsafe_allow_html=True)

        # Create columns for the query input and any additional controls
        query_col, button_col = st.columns([4, 1])

        with query_col:
            query = st.text_input("💬 Ask a question about the documents:")

        with button_col:
            search_button = st.button("Search", key="search_button")

    # Process query when button is clicked or Enter is pressed in the text input
    if query and (search_button or not st.session_state.get('last_query') == query):
        st.session_state.last_query = query

        # Show query processing status
        query_status = st.empty()
        query_status.markdown(f"<div class='status-box info-box'>⏳ Processing query: '{query}'</div>", unsafe_allow_html=True)

        try:
            # Build filter based on selected filters
            filter_dict = build_filter_dict()

            # Show what filters are being applied
            if filter_dict:
                filter_display = ", ".join([f"{k}: {v}" for k, v in filter_dict.items()])
                show_status(f"Applying filters: {filter_display}", "info")

            # Set up retriever with filters
            retriever = st.session_state.vector.as_retriever(
                search_kwargs={"filter": filter_dict}
            )
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            with st.spinner("Thinking..."):
                try:
                    # Get response
                    response = retrieval_chain.invoke({"input": query})

                    # Clear status message
                    query_status.empty()

                    # Display answer
                    st.markdown("### 🧠 Answer")
                    st.write(response["answer"])

                    # Add to chat history
                    chat_entry = {
                        "question": query,
                        "answer": response["answer"],
                        "context_sources": [chunk.metadata.get("file_name", "Unknown") for chunk in response["context"]],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.chat_history.append(chat_entry)
                    save_chat_history(st.session_state.chat_history)
                    show_status("Answer generated and saved to chat history", "success")

                    # Group context by document source
                    sources_dict = {}
                    for chunk in response["context"]:
                        source = chunk.metadata.get("file_name", "Unknown source")
                        if source not in sources_dict:
                            sources_dict[source] = []
                        sources_dict[source].append(chunk)

                    # Display context used
                    with st.expander("📄 Context Used", expanded=True):
                        st.markdown(f"**Sources used:** {len(sources_dict)} document(s)")
                        for source, chunks in sources_dict.items():
                            st.subheader(f"Source: {source}")
                            for i, chunk in enumerate(chunks):
                                with st.container():
                                    st.markdown(f"**Chunk {i+1}** | **Page:** {chunk.metadata.get('page', 'N/A')} | **Type:** {chunk.metadata.get('file_type', 'N/A')}")
                                    st.markdown(chunk.page_content)
                                    st.markdown("---")

                    # Document comparison section
                    if len(sources_dict) > 1:
                        with st.expander("📊 Document Comparison", expanded=True):
                            st.markdown("### Information from different sources")
                            st.markdown("This section shows how information compares across different documents.")

                            # Create columns for each source (up to 3 columns)
                            cols = st.columns(min(len(sources_dict), 3))

                            # Display content from each source in its own column
                            for i, (source, chunks) in enumerate(sources_dict.items()):
                                col_idx = i % len(cols)
                                with cols[col_idx]:
                                    st.markdown(f"**{source}**")
                                    for chunk in chunks[:2]:  # Limit to first 2 chunks per source
                                        st.markdown(f"*{chunk.metadata.get('file_type', 'unknown type')}*")
                                        st.markdown(chunk.page_content[:200] + "...")
                                        st.markdown("---")

                    # Export answer button
                    st.markdown("### Export Results")
                    export_answer(query, response["answer"], response["context"])

                except Exception as query_error:
                    query_status.empty()
                    show_status(f"Error generating answer: {query_error}", "error")
                    st.exception(query_error)
        except Exception as e:
            query_status.empty()
            show_status(f"Error setting up retrieval: {e}", "error")
            st.exception(e)
    elif not "vector" in st.session_state:
        show_status("Please upload a document first to start asking questions.", "info")
