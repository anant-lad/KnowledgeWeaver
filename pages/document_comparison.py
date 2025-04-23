import streamlit as st
import os
import tempfile
import base64
import json
import uuid
import shutil
from datetime import datetime
from dotenv import load_dotenv
import difflib

# Import authentication utilities
from utils.auth_utils import initialize_auth_state, get_current_user, logout_user
from utils.auth_components import render_auth_page

# Import OpenAI for analysis
from openai import OpenAI

# Import LangChain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.documents import Document

# Try to import image processing libraries
try:
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False
    st.error("PIL/Pillow is required for image analysis. Please install it with: pip install Pillow")

# Try to import OCR libraries
OCR_SUPPORT = False
try:
    import pytesseract
    import cv2
    import numpy as np
    OCR_SUPPORT = True
except ImportError:
    st.warning("OCR support requires additional libraries. Install them with: pip install pytesseract opencv-python numpy")
    st.info("You'll also need to install Tesseract OCR on your system: https://github.com/tesseract-ocr/tesseract")

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
neo4j_url = os.getenv("NEO4J_URI")  # Using NEO4J_URI from .env file
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Initialize OpenAI client
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    st.error("OpenAI API key not found. Please add it to your .env file as OPENAI_API_KEY.")
    client = None

# Page configuration
st.set_page_config(page_title="Document Comparison - KnowledgeWeaver", layout="wide")

# Initialize authentication state
initialize_auth_state()

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
    .image-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    .image-container img {
        max-width: 100%;
        max-height: 500px;
    }
    .analysis-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .comparison-highlight {
        background-color: #e6f7ff;
        padding: 0.25rem;
        border-radius: 0.25rem;
    }
    .difference-highlight {
        background-color: #ffebee;
        padding: 0.25rem;
        border-radius: 0.25rem;
    }
    .similarity-highlight {
        background-color: #e8f5e9;
        padding: 0.25rem;
        border-radius: 0.25rem;
    }
    .file-card {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .comparison-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
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

# Function to get image base64 encoding
def get_image_base64(image_path):
    """Convert an image to base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to analyze image with OpenAI Vision
def analyze_image_with_openai(image_path, prompt):
    """Analyze an image using OpenAI's Vision model"""
    if client is None:
        st.error("Cannot analyze image: OpenAI client is not initialized. Please check your API key.")
        return None

    try:
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Call OpenAI API with the current GPT-4 Vision model
        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to use gpt-4o which supports vision
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )

        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return None

# Function to perform OCR on an image
def perform_ocr(image_path, lang='eng', preprocess=None):
    """Extract text from an image using OCR

    Args:
        image_path: Path to the image file
        lang: Language for OCR (default: 'eng')
        preprocess: Preprocessing method (None, 'thresh', 'blur', or 'adaptive')

    Returns:
        Extracted text as a string
    """
    if not OCR_SUPPORT:
        return "OCR is not available. Please install the required libraries."

    try:
        # Read the image directly into memory using PIL instead of OpenCV
        # This avoids file locking issues that can occur with OpenCV
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()

        # Convert bytes to numpy array
        nparr = np.frombuffer(img_data, np.uint8)

        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return f"Error: Could not decode image data"

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply preprocessing if specified
        if preprocess == 'thresh':
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif preprocess == 'blur':
            gray = cv2.medianBlur(gray, 3)
        elif preprocess == 'adaptive':
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Perform OCR directly on the image array
        # Convert OpenCV image to PIL Image for pytesseract
        pil_img = Image.fromarray(gray)
        text = pytesseract.image_to_string(pil_img, lang=lang)

        return text
    except Exception as e:
        return f"Error performing OCR: {e}"

# Function to extract text from a document
def extract_text_from_document(file_path, file_extension):
    """Extract text from a document based on its file type"""
    documents = []

    try:
        # Check if file exists and is accessible
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return documents

        if file_extension == ".pdf":
            try:
                loader = PyMuPDFLoader(file_path)
                documents = loader.load()
                st.success(f"Successfully extracted {len(documents)} pages from PDF")
            except Exception as pdf_error:
                st.error(f"Error loading PDF: {pdf_error}")

        elif file_extension == ".docx":
            try:
                loader = UnstructuredWordDocumentLoader(file_path)
                documents = loader.load()
                st.success(f"Successfully extracted content from Word document")
            except Exception as docx_error:
                st.error(f"Error loading Word document: {docx_error}")

        elif file_extension in [".jpg", ".jpeg", ".png"] and IMAGE_SUPPORT:
            # For images, use OCR or AI-based extraction
            extraction_method = st.session_state.get('extraction_method', 'AI-based')
            extracted_text = ""

            if extraction_method == 'AI-based':
                # Use GPT-4 Vision for extraction
                extraction_prompt = "Please extract all text visible in this image. Format it properly maintaining paragraphs, bullet points, and tables if present."
                extracted_text = analyze_image_with_openai(file_path, extraction_prompt)
                if extracted_text:
                    st.success(f"Successfully extracted text using AI-based method")
                else:
                    st.warning("AI-based extraction returned no text")
            else:
                # Use OCR for extraction
                if OCR_SUPPORT:
                    lang = st.session_state.get('ocr_lang', 'eng')
                    preprocess = st.session_state.get('preprocess_param', None)

                    # Show what method is being used
                    preprocess_name = preprocess if preprocess else "None"
                    st.info(f"Method: OCR (Tesseract) with {preprocess_name} preprocessing, language: {lang}")

                    # Use a completely different approach to avoid file access issues
                    try:
                        # Load the image directly with PIL
                        pil_img = Image.open(file_path)

                        # Convert to grayscale if needed
                        if pil_img.mode != 'L':
                            pil_img = pil_img.convert('L')

                        # Apply preprocessing if specified (using PIL instead of OpenCV)
                        if preprocess == 'thresh':
                            # Use PIL's point function for thresholding
                            pil_img = pil_img.point(lambda x: 0 if x < 128 else 255, '1')
                        elif preprocess == 'blur':
                            # Use PIL's filter for blurring
                            pil_img = pil_img.filter(ImageFilter.MedianFilter(3))
                        elif preprocess == 'adaptive':
                            # For adaptive, we'll use a simple enhancement as PIL doesn't have direct adaptive threshold
                            enhancer = ImageEnhance.Contrast(pil_img)
                            pil_img = enhancer.enhance(2.0)  # Increase contrast

                        # Perform OCR directly on the PIL image
                        extracted_text = pytesseract.image_to_string(pil_img, lang=lang)

                        if extracted_text.strip():
                            st.success(f"Successfully extracted text using OCR")
                        else:
                            st.warning("OCR extraction returned no text")
                    except Exception as ocr_error:
                        error_msg = f"Error performing OCR: {ocr_error}"
                        st.error(error_msg)
                        extracted_text = error_msg
                else:
                    extracted_text = "OCR is not available. Please install the required libraries."
                    st.error(extracted_text)

            # Create a document with the extracted text
            documents = [Document(page_content=extracted_text, metadata={"source": os.path.basename(file_path)})]
    except Exception as e:
        st.error(f"Error extracting text from document: {e}")

    return documents

# Function to compare two texts and highlight differences
def compare_texts(text1, text2):
    """Compare two texts and return HTML with highlighted differences"""
    # Split texts into lines
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    # Use difflib to compare lines
    d = difflib.Differ()
    diff = list(d.compare(lines1, lines2))

    # Format the differences
    html_output = []
    for line in diff:
        if line.startswith('+ '):
            html_output.append(f"<div class='difference-highlight'>+ {line[2:]}</div>")
        elif line.startswith('- '):
            html_output.append(f"<div class='difference-highlight'>- {line[2:]}</div>")
        elif line.startswith('? '):
            continue  # Skip the indicator line
        else:
            html_output.append(f"<div class='similarity-highlight'>{line[2:]}</div>")

    return "<br>".join(html_output)

# Function to save comparison results
def save_comparison_results(file_names, texts, comparison_result, query=None, answer=None):
    """Save comparison results to a JSON file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create results directory if it doesn't exist
    if not os.path.exists("comparison_results"):
        os.makedirs("comparison_results")

    # Create a unique filename
    filename = f"comparison_results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Prepare the data
    data = {
        "file_names": file_names,
        "texts": texts,
        "comparison_result": comparison_result,
        "timestamp": timestamp
    }

    # Add query and answer if available
    if query:
        data["query"] = query
    if answer:
        data["answer"] = answer

    # Save to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    return filename

# Function to estimate token count
def estimate_tokens(text):
    """Estimate the number of tokens in a text (rough approximation)"""
    # A very rough approximation: 1 token ~= 4 characters for English text
    return len(text) // 4

# Function to truncate text to fit within token limits
def truncate_for_token_limit(texts, file_names, max_tokens=6000):
    """Truncate texts to fit within token limits"""
    # Reserve tokens for the prompt template and other text
    reserved_tokens = 1000
    available_tokens = max_tokens - reserved_tokens

    # Calculate total tokens in all texts
    total_tokens = sum(estimate_tokens(text) for text in texts)

    # If we're under the limit, return the original texts
    if total_tokens <= available_tokens:
        return texts, file_names

    # Calculate how much we need to reduce each text (proportionally)
    reduction_factor = available_tokens / total_tokens

    # Truncate each text proportionally
    truncated_texts = []
    for text in texts:
        tokens = estimate_tokens(text)
        target_tokens = int(tokens * reduction_factor)
        # Convert back to approximate character count
        target_chars = target_tokens * 4

        if len(text) > target_chars:
            truncated = text[:target_chars] + "... [truncated due to length]"
        else:
            truncated = text

        truncated_texts.append(truncated)

    return truncated_texts, file_names

# Function to analyze and compare documents using OpenAI
def analyze_and_compare_documents(texts, file_names, query=None):
    """Analyze and compare documents using OpenAI"""
    if client is None:
        st.error("Cannot analyze documents: OpenAI client is not initialized. Please check your API key.")
        return None

    try:
        # Truncate texts to fit within token limits
        truncated_texts, file_names = truncate_for_token_limit(texts, file_names)

        # Create document summaries for the prompt
        document_contents = []
        for i, text in enumerate(truncated_texts):
            # Limit each document's text in the prompt
            max_chars = min(len(text), 2000)  # Limit to 2000 characters per document in the prompt
            document_contents.append(f'Document {i+1} ({file_names[i]}): {text[:max_chars]}' +
                                    ('...' if len(text) > max_chars else ''))

        # Create a prompt for comparison
        if query:
            prompt = f"""
            I have the following documents:

            {', '.join(file_names)}

            Here are their contents (some may be truncated due to length):

            {' '.join(document_contents)}

            Please answer this question about these documents: {query}

            In your answer:
            1. Provide a direct answer to the question
            2. Compare the information across the documents
            3. Highlight any similarities and differences
            4. Summarize the key points from each document
            """
        else:
            prompt = f"""
            I have the following documents:

            {', '.join(file_names)}

            Here are their contents (some may be truncated due to length):

            {' '.join(document_contents)}

            Please compare these documents and provide:
            1. A summary of each document
            2. Key similarities between the documents
            3. Important differences between the documents
            4. An overall analysis of how these documents relate to each other
            """

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that specializes in document comparison and analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )

        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing documents: {e}")
        return None

# Function to load chat history from disk
def load_comparison_chat_history():
    try:
        if os.path.exists('comparison_chat_history.json'):
            with open('comparison_chat_history.json', 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading comparison chat history: {e}")
        return []

# Function to save chat history to disk
def save_comparison_chat_history(history):
    try:
        with open('comparison_chat_history.json', 'w') as f:
            json.dump(history, f)
    except Exception as e:
        st.error(f"Error saving comparison chat history: {e}")

# Initialize session state for document comparison
if 'comparison_documents' not in st.session_state:
    st.session_state.comparison_documents = []
if 'comparison_file_paths' not in st.session_state:
    st.session_state.comparison_file_paths = []
if 'comparison_file_names' not in st.session_state:
    st.session_state.comparison_file_names = []
if 'comparison_texts' not in st.session_state:
    st.session_state.comparison_texts = []
if 'comparison_result' not in st.session_state:
    st.session_state.comparison_result = None
if 'qa_result' not in st.session_state:
    st.session_state.qa_result = None
if 'comparison_chat_history' not in st.session_state:
    st.session_state.comparison_chat_history = load_comparison_chat_history()
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# No login/signup buttons in the main content area

# Check if user is authenticated
if not st.session_state.authenticated:
    # Show login/signup page in sidebar
    auth_success = render_auth_page(neo4j_url, neo4j_user, neo4j_password)

    # If not authenticated, don't show the rest of the app
    if not auth_success:
        st.stop()

# Show navigation links for authenticated users
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
    <a href='/document_comparison' style='text-decoration: none;'>
        <div style='background-color: #9C27B0; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; text-align: center;'>
            📊 Document Comparison
        </div>
    </a>
</div>
""", unsafe_allow_html=True)

# Main page content (only shown to authenticated users)
st.title("📊 Document & Image Comparison")
st.markdown("Upload documents and images to compare them and ask questions")

# Create sidebar for login/signup
sidebar = st.sidebar
sidebar.title("📋 Authentication")

# Add login/signup buttons to sidebar
user = get_current_user()
if user:
    # User is logged in
    sidebar.markdown(f"### 👤 {user['username']}")
    sidebar.success("Your data will be saved permanently.")

    # Admin panel is only accessible by direct URL - no UI indication

    # Logout button
    if sidebar.button("🚪 Logout"):
        logout_user()
        st.rerun()
else:
    # User is not logged in
    sidebar.warning("Anonymous Mode: Data will not be saved permanently")
    if sidebar.button("🔑 Login"):
        st.session_state.show_auth_popup = True
        st.session_state.auth_popup_mode = "login"
        st.rerun()
    if sidebar.button("✏️ Sign Up"):
        st.session_state.show_auth_popup = True
        st.session_state.auth_popup_mode = "signup"
        st.rerun()

# Add information about the app
sidebar.markdown("---")
sidebar.markdown("## ℹ️ About")
sidebar.markdown("""
This application allows you to:
- Compare multiple documents
- Extract text from images
- Ask questions about documents
- Analyze similarities and differences
""")

# Add chat history to sidebar
sidebar.markdown("---")
sidebar.markdown("## 💬 Chat History")

# Create a dropdown for chat history
if 'show_comparison_chat_history' not in st.session_state:
    st.session_state.show_comparison_chat_history = False

if sidebar.button('Show/Hide Chat History', key='toggle_comparison_chat_history'):
    st.session_state.show_comparison_chat_history = not st.session_state.show_comparison_chat_history

# Display recent chat history (limited to 10 entries)
if st.session_state.show_comparison_chat_history:
    # Get current user
    current_user = get_current_user()

    # Only show chat history for authenticated users
    if current_user:
        # Filter chat history for current user
        user_history = [entry for entry in st.session_state.comparison_chat_history
                       if entry.get('user_id') == current_user.get('user_id')]

        if user_history:
            # Sort by timestamp in descending order (newest first)
            sorted_history = sorted(user_history,
                                   key=lambda x: x.get('timestamp', ''),
                                   reverse=True)

            # Display the 10 most recent entries
            for i, entry in enumerate(sorted_history[:10]):
                sidebar.markdown(f"**Q: {entry['question'][:50]}{'...' if len(entry['question']) > 50 else ''}**")
                sidebar.markdown(f"A: {entry['answer'][:100]}{'...' if len(entry['answer']) > 100 else ''}")
                if 'documents' in entry:
                    doc_names = [os.path.basename(doc) for doc in entry.get('documents', [])[:2]]
                    sidebar.markdown(f"Sources: {', '.join(doc_names)}")
                sidebar.markdown(f"Time: {entry.get('timestamp', 'Unknown')}")
                sidebar.markdown("---")

            # Add a button to clear user's chat history
            if sidebar.button("🗑️ Clear My Chat History", key="clear_comparison_history"):
                # Remove only this user's entries from chat history
                st.session_state.comparison_chat_history = [entry for entry in st.session_state.comparison_chat_history
                                                          if entry.get('user_id') != current_user.get('user_id')]
                save_comparison_chat_history(st.session_state.comparison_chat_history)
                st.rerun()
        else:
            sidebar.info("No chat history available yet. Ask questions to see them appear here.")
    else:
        sidebar.info("Please log in to view your chat history.")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Document Comparison", "Q&A"])

with tab1:
    st.header("Compare Documents and Images")

    # File uploader for comparison
    uploaded_files = st.file_uploader(
        "Upload documents and images to compare",
        type=["pdf", "docx", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="comparison_files"
    )

    # Text extraction options for images
    if any(f.name.lower().endswith(('.jpg', '.jpeg', '.png')) for f in uploaded_files if f):
        st.subheader("Text Extraction Options for Images")

        extraction_col1, extraction_col2 = st.columns(2)

        with extraction_col1:
            extraction_method = st.radio(
                "Extraction Method",
                options=["AI-based", "OCR"],
                index=0,
                key="extraction_method",
                help="Choose between AI-based extraction or traditional OCR"
            )
            st.session_state.extraction_method = extraction_method

        with extraction_col2:
            if extraction_method == "OCR" and OCR_SUPPORT:
                ocr_lang = st.selectbox(
                    "OCR Language",
                    options=["eng", "fra", "deu", "spa", "ita", "por", "rus", "jpn", "kor", "chi_sim"],
                    index=0,
                    format_func=lambda x: {
                        "eng": "English", "fra": "French", "deu": "German",
                        "spa": "Spanish", "ita": "Italian", "por": "Portuguese",
                        "rus": "Russian", "jpn": "Japanese", "kor": "Korean", "chi_sim": "Chinese (Simplified)"
                    }.get(x, x),
                    key="ocr_lang"
                )
                st.session_state.ocr_lang = ocr_lang

                preprocessing = st.selectbox(
                    "Image Preprocessing",
                    options=["None", "Threshold", "Blur", "Adaptive Threshold"],
                    index=0,
                    format_func=lambda x: x if x != "None" else "No Preprocessing",
                    key="preprocessing"
                )

                # Map preprocessing options to function parameters
                preprocess_map = {
                    "None": None,
                    "Threshold": "thresh",
                    "Blur": "blur",
                    "Adaptive Threshold": "adaptive"
                }
                st.session_state.preprocess_param = preprocess_map[preprocessing]

    # Process button
    if uploaded_files and len(uploaded_files) >= 2:
        if st.button("Process and Compare Documents", key="process_comparison"):
            # Clear previous results
            st.session_state.comparison_documents = []
            st.session_state.comparison_file_paths = []
            st.session_state.comparison_file_names = []
            st.session_state.comparison_texts = []
            st.session_state.comparison_result = None

            # Process each uploaded file
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    # Save the uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    # Get file extension
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

                    # Extract text from the document
                    show_status(f"Extracting text from {uploaded_file.name}...", "info")
                    documents = extract_text_from_document(tmp_path, file_extension)

                    if documents:
                        # Add to our lists
                        st.session_state.comparison_documents.extend(documents)
                        st.session_state.comparison_file_paths.append(tmp_path)
                        st.session_state.comparison_file_names.append(uploaded_file.name)

                        # Extract text content
                        text_content = "\n\n".join([doc.page_content for doc in documents])
                        st.session_state.comparison_texts.append(text_content)

                        show_status(f"Successfully processed {uploaded_file.name}", "success")
                    else:
                        show_status(f"No content could be extracted from {uploaded_file.name}", "warning")

                # Analyze and compare the documents
                if len(st.session_state.comparison_texts) >= 2:
                    show_status("Analyzing and comparing documents...", "info")

                    # Check if documents might be too large
                    total_chars = sum(len(text) for text in st.session_state.comparison_texts)
                    if total_chars > 30000:  # Roughly 7,500 tokens
                        show_status(f"Documents are large ({total_chars:,} characters). Some content may be truncated for analysis.", "warning")

                    comparison_result = analyze_and_compare_documents(
                        st.session_state.comparison_texts,
                        st.session_state.comparison_file_names
                    )

                    if comparison_result:
                        st.session_state.comparison_result = comparison_result
                        show_status("Comparison completed successfully!", "success")
                    else:
                        show_status("Failed to generate comparison", "error")
                else:
                    show_status("Not enough valid documents to compare", "warning")
    elif uploaded_files:
        st.warning("Please upload at least 2 documents to compare")

    # Display comparison results
    if st.session_state.comparison_result:
        st.subheader("Comparison Results")

        # Display the files side by side
        num_files = len(st.session_state.comparison_file_names)
        cols = st.columns(min(num_files, 3))

        for i, (file_name, file_path, text) in enumerate(zip(
            st.session_state.comparison_file_names,
            st.session_state.comparison_file_paths,
            st.session_state.comparison_texts
        )):
            col_idx = i % len(cols)
            with cols[col_idx]:
                st.markdown(f"**{file_name}**")

                # Display image if it's an image file
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    st.image(file_path, caption=file_name, use_column_width=True)

                # Display text preview
                st.markdown("**Text Content:**")
                st.markdown(f"<div style='max-height: 200px; overflow-y: auto;'>{text[:500]}{'...' if len(text) > 500 else ''}</div>", unsafe_allow_html=True)

        # Display the comparison analysis
        st.markdown("### Analysis")
        st.markdown(st.session_state.comparison_result)

        # Text comparison for two documents
        if len(st.session_state.comparison_texts) == 2:
            with st.expander("Detailed Text Comparison", expanded=False):
                st.markdown("This view highlights similarities and differences between the two documents:")
                html_diff = compare_texts(st.session_state.comparison_texts[0], st.session_state.comparison_texts[1])
                st.markdown(html_diff, unsafe_allow_html=True)

        # Save results button
        if st.button("Save Comparison Results", key="save_comparison"):
            result_file = save_comparison_results(
                st.session_state.comparison_file_names,
                st.session_state.comparison_texts,
                st.session_state.comparison_result
            )
            show_status(f"Comparison results saved to {result_file}", "success")

            # Download button
            try:
                with open(result_file, "r") as f:
                    st.download_button(
                        label="📥 Download Comparison Results",
                        data=f.read(),
                        file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_comparison"
                    )
            except Exception as download_error:
                st.error(f"Error creating download button: {download_error}")

with tab2:
    st.header("Ask Questions About Documents")

    if not st.session_state.comparison_texts:
        st.info("Please upload and process documents in the 'Document Comparison' tab first")
    else:
        st.markdown(f"You have {len(st.session_state.comparison_file_names)} documents loaded:")
        for file_name in st.session_state.comparison_file_names:
            st.markdown(f"- {file_name}")

        # Query input
        query = st.text_area("Enter your question about the documents:", height=100)

        if query and st.button("Ask Question", key="ask_question"):
            with st.spinner("Analyzing documents and generating answer..."):
                # Check if documents might be too large
                total_chars = sum(len(text) for text in st.session_state.comparison_texts)
                if total_chars > 30000:  # Roughly 7,500 tokens
                    show_status(f"Documents are large ({total_chars:,} characters). Some content may be truncated for analysis.", "warning")

                qa_result = analyze_and_compare_documents(
                    st.session_state.comparison_texts,
                    st.session_state.comparison_file_names,
                    query
                )

                if qa_result:
                    st.session_state.qa_result = qa_result
                    st.session_state.last_query = query

                    # Add to chat history with user ID
                    user = get_current_user()
                    chat_entry = {
                        "question": query,
                        "answer": qa_result,
                        "documents": st.session_state.comparison_file_names,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "user_id": user["user_id"] if user else None
                    }
                    st.session_state.comparison_chat_history.append(chat_entry)
                    save_comparison_chat_history(st.session_state.comparison_chat_history)

                    show_status("Answer generated successfully and saved to chat history!", "success")
                else:
                    show_status("Failed to generate answer", "error")

        # Display Q&A results
        if st.session_state.qa_result:
            st.subheader(f"Answer to: {st.session_state.get('last_query', 'your question')}")
            st.markdown(st.session_state.qa_result)

            # Display source information
            with st.expander("📄 Source Documents", expanded=True):
                st.markdown(f"**Sources used:** {len(st.session_state.comparison_file_names)} document(s)")
                for i, file_name in enumerate(st.session_state.comparison_file_names):
                    st.markdown(f"**{i+1}. {file_name}**")
                    # Display image if it's an image file
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')) and i < len(st.session_state.comparison_file_paths):
                        st.image(st.session_state.comparison_file_paths[i], width=200)
                    # Display text preview
                    if i < len(st.session_state.comparison_texts):
                        st.markdown(f"*Preview:* {st.session_state.comparison_texts[i][:200]}...")

            # Save results button
            if st.button("Save Q&A Results", key="save_qa"):
                result_file = save_comparison_results(
                    st.session_state.comparison_file_names,
                    st.session_state.comparison_texts,
                    st.session_state.comparison_result,
                    st.session_state.get('last_query', ''),
                    st.session_state.qa_result
                )
                show_status(f"Q&A results saved to {result_file}", "success")

                # Download button
                try:
                    with open(result_file, "r") as f:
                        st.download_button(
                            label="📥 Download Q&A Results",
                            data=f.read(),
                            file_name=f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key="download_qa"
                        )
                except Exception as download_error:
                    st.error(f"Error creating download button: {download_error}")

# Footer
st.markdown("---")
st.markdown("### 📊 Comparison History")

# Create a row of buttons for clearing data
col1, col2 = st.columns(2)

# Add a button to clear session data
with col1:
    if st.button("Clear Session Data", key="clear_session"):
        # Clear all session data related to document comparison
        st.session_state.comparison_documents = []
        st.session_state.comparison_file_paths = []
        st.session_state.comparison_file_names = []
        st.session_state.comparison_texts = []
        st.session_state.comparison_result = None
        st.session_state.qa_result = None
        st.session_state.session_id = str(uuid.uuid4())
        show_status("Session data cleared. You can upload new documents.", "success")
        st.rerun()

# Add a button to clear analysis history
with col2:
    if st.button("Clear Analysis History", key="clear_analysis_history"):
        # Remove all comparison result files
        if os.path.exists("comparison_results"):
            try:
                comparison_files = [f for f in os.listdir("comparison_results") if f.endswith(".json")]
                for file in comparison_files:
                    try:
                        os.remove(os.path.join("comparison_results", file))
                    except Exception as e:
                        st.error(f"Error removing file {file}: {e}")
                show_status("Analysis history cleared successfully!", "success")
            except Exception as e:
                st.error(f"Error clearing analysis history: {e}")
        else:
            st.info("No analysis history to clear.")
        st.rerun()

# Create comparison_results directory if it doesn't exist
if not os.path.exists("comparison_results"):
    try:
        os.makedirs("comparison_results")
        show_status("Created comparison_results directory", "info")
    except Exception as e:
        st.error(f"Error creating comparison_results directory: {e}")

# Display previous comparisons if available
if os.path.exists("comparison_results"):
    try:
        comparison_files = [f for f in os.listdir("comparison_results") if f.endswith(".json")]
    except Exception as e:
        st.error(f"Error reading comparison_results directory: {e}")
        comparison_files = []

    if comparison_files:
        comparison_files.sort(reverse=True)  # Most recent first

        for i, file in enumerate(comparison_files[:5]):  # Show only the 5 most recent
            try:
                with open(os.path.join("comparison_results", file), "r") as f:
                    try:
                        data = json.load(f)
                        with st.expander(f"Comparison {i+1}: {', '.join(data.get('file_names', []))} - {data.get('timestamp', 'Unknown date')}"):
                            if 'query' in data and 'answer' in data:
                                st.markdown(f"**Question:** {data.get('query', 'No question')}")
                                st.markdown(f"**Answer:** {data.get('answer', 'No answer')}")
                            else:
                                st.markdown(f"**Comparison Result:** {data.get('comparison_result', 'No comparison result')}")
                    except json.JSONDecodeError as json_error:
                        st.error(f"Error parsing JSON in file {file}: {json_error}")
                    except Exception as e:
                        st.error(f"Error processing comparison file {file}: {e}")
            except FileNotFoundError:
                st.warning(f"File {file} not found. It may have been moved or deleted.")
            except Exception as e:
                st.error(f"Error opening comparison file {file}: {e}")
    else:
        st.info("No previous comparisons found.")
else:
    st.info("No comparison history available yet.")
