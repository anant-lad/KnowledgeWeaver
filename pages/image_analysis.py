import streamlit as st
import os
import tempfile
import base64
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Import authentication utilities
from utils.auth_utils import initialize_auth_state, get_current_user, logout_user
from utils.auth_components import render_auth_page

# Import OpenAI for image analysis
from openai import OpenAI

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
neo4j_url = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Initialize OpenAI client
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    st.error("OpenAI API key not found. Please add it to your .env file as OPENAI_API_KEY.")
    client = None

# Page configuration
st.set_page_config(page_title="Image Analysis - KnowledgeWeaver", layout="wide")

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
        # Read the image with OpenCV
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply preprocessing if specified
        if preprocess == 'thresh':
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif preprocess == 'blur':
            gray = cv2.medianBlur(gray, 3)
        elif preprocess == 'adaptive':
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Save the preprocessed image to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(temp_file.name, gray)

        # Perform OCR
        text = pytesseract.image_to_string(Image.open(temp_file.name), lang=lang)

        # Clean up
        os.unlink(temp_file.name)

        return text
    except Exception as e:
        return f"Error performing OCR: {e}"

# Function to load image analysis chat history from disk
def load_image_chat_history():
    try:
        if os.path.exists('image_chat_history.json'):
            with open('image_chat_history.json', 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading image chat history: {e}")
        return []

# Function to save image analysis chat history to disk
def save_image_chat_history(history):
    try:
        with open('image_chat_history.json', 'w') as f:
            json.dump(history, f)
    except Exception as e:
        st.error(f"Error saving image chat history: {e}")

# Function to save analysis results
def save_analysis_results(image_name, prompt, analysis, image_path=None):
    """Save analysis results to a JSON file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create results directory if it doesn't exist
    if not os.path.exists("analysis_results"):
        os.makedirs("analysis_results")

    # Create a unique filename
    filename = f"analysis_results/image_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Prepare the data
    data = {
        "image_name": image_name,
        "prompt": prompt,
        "analysis": analysis,
        "timestamp": timestamp
    }

    # Add image base64 if available
    if image_path:
        try:
            data["image_base64"] = get_image_base64(image_path)
        except Exception:
            pass

    # Save to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    return filename

# Initialize session state for image analysis
if 'image_chat_history' not in st.session_state:
    st.session_state.image_chat_history = load_image_chat_history()
if 'image_session_id' not in st.session_state:
    st.session_state.image_session_id = str(uuid.uuid4())
if 'current_images' not in st.session_state:
    st.session_state.current_images = []

# No login/signup buttons in the main content area

# Check if user is authenticated
if not st.session_state.authenticated:
    # Show login/signup page in sidebar
    auth_success = render_auth_page(neo4j_url, neo4j_user, neo4j_password)

    # If not authenticated, don't show the rest of the app
    if not auth_success:
        st.stop()

# Main page content (only shown to authenticated users)
st.title("🖼️ Image Analysis")
st.markdown("Upload images for detailed AI-powered analysis")

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
- Analyze images with AI
- Extract text from images
- Detect objects in images
- Save and download analysis results
""")

# Add chat history to sidebar
sidebar.markdown("---")
sidebar.markdown("## 💬 Chat History")

# Create a dropdown for chat history
if 'show_image_chat_history' not in st.session_state:
    st.session_state.show_image_chat_history = False

if sidebar.button('Show/Hide Chat History', key='toggle_image_chat_history'):
    st.session_state.show_image_chat_history = not st.session_state.show_image_chat_history

# Display recent chat history (limited to 10 entries)
if st.session_state.show_image_chat_history:
    # Get current user
    current_user = get_current_user()

    # Only show chat history for authenticated users
    if current_user:
        # Filter chat history for current user
        user_history = [entry for entry in st.session_state.image_chat_history
                       if entry.get('user_id') == current_user.get('user_id')]

        if user_history:
            # Sort by timestamp in descending order (newest first)
            sorted_history = sorted(user_history,
                                   key=lambda x: x.get('timestamp', ''),
                                   reverse=True)

            # Display the 10 most recent entries
            for i, entry in enumerate(sorted_history[:10]):
                sidebar.markdown(f"**Image: {entry.get('image_name', 'Unknown')[:30]}{'...' if len(entry.get('image_name', 'Unknown')) > 30 else ''}**")
                sidebar.markdown(f"Type: {entry.get('type', 'Unknown analysis')}")
                sidebar.markdown(f"Prompt: {entry.get('prompt', 'No prompt')[:50]}{'...' if len(entry.get('prompt', 'No prompt')) > 50 else ''}")
                sidebar.markdown(f"Analysis: {entry.get('analysis', 'No analysis')[:100]}{'...' if len(entry.get('analysis', 'No analysis')) > 100 else ''}")
                sidebar.markdown(f"Time: {entry.get('timestamp', 'Unknown')}")
                sidebar.markdown("---")

            # Add a button to clear user's chat history
            if sidebar.button("🗑️ Clear My Chat History", key="clear_image_history"):
                # Remove only this user's entries from chat history
                st.session_state.image_chat_history = [entry for entry in st.session_state.image_chat_history
                                                     if entry.get('user_id') != current_user.get('user_id')]
                save_image_chat_history(st.session_state.image_chat_history)
                st.rerun()
        else:
            sidebar.info("No chat history available yet. Analyze images to see results appear here.")
    else:
        sidebar.info("Please log in to view your chat history.")

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

# Create tabs for different analysis types
tab1, tab2, tab3 = st.tabs(["General Analysis", "Text Extraction", "Object Detection"])

with tab1:
    st.header("General Image Analysis")

    # File uploader with multiple file support
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="general_analysis")

    if uploaded_files:
        # Analysis options that apply to all images
        analysis_prompt = st.text_area(
            "Analysis Prompt",
            value="Please analyze this image in detail. Describe what you see, including objects, people, colors, and the overall scene.",
            height=100
        )

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Create a container for each image
            image_container = st.container()
            with image_container:
                st.markdown(f"### Processing: {uploaded_file.name}")

                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Display the image
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(tmp_path, caption=uploaded_file.name)
                st.markdown("</div>", unsafe_allow_html=True)

                # Analysis button for this specific image
                if st.button(f"Analyze Image", key=f"analyze_general_{uploaded_file.name}"):
                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        analysis_result = analyze_image_with_openai(tmp_path, analysis_prompt)

                        if analysis_result:
                            st.markdown("<div class='analysis-result'>", unsafe_allow_html=True)
                            st.markdown("### Analysis Result")
                            st.markdown(analysis_result)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Save results
                            result_file = save_analysis_results(uploaded_file.name, analysis_prompt, analysis_result, tmp_path)

                            # Add to chat history with user ID
                            user = get_current_user()
                            chat_entry = {
                                "image_name": uploaded_file.name,
                                "prompt": analysis_prompt,
                                "analysis": analysis_result,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "type": "general_analysis",
                                "user_id": user["user_id"] if user else None
                            }
                            st.session_state.image_chat_history.append(chat_entry)
                            save_image_chat_history(st.session_state.image_chat_history)

                            # Add to current session images
                            if uploaded_file.name not in [img["name"] for img in st.session_state.current_images]:
                                st.session_state.current_images.append({
                                    "name": uploaded_file.name,
                                    "path": tmp_path,
                                    "type": "general_analysis"
                                })

                            show_status(f"Analysis saved to {result_file} and added to history", "success")

                            # Download button
                            try:
                                with open(result_file, "r") as f:
                                    st.download_button(
                                        label="📥 Download Analysis",
                                        data=f.read(),
                                        file_name=f"analysis_{uploaded_file.name.split('.')[0]}.json",
                                        mime="application/json",
                                        key=f"download_analysis_{uploaded_file.name}"
                                    )
                            except Exception as download_error:
                                st.error(f"Error creating download button: {download_error}")

with tab2:
    st.header("Text Extraction from Images")

    # File uploader with multiple file support
    uploaded_files = st.file_uploader("Upload images containing text", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="text_extraction")

    if uploaded_files:
        # Extraction options that apply to all images
        extraction_method = st.radio(
            "Extraction Method",
            options=["AI-based (GPT-4 Vision)", "OCR (Tesseract)"],
            index=0,
            help="Choose between AI-based extraction or traditional OCR"
        )

        if extraction_method == "OCR (Tesseract)" and OCR_SUPPORT:
            # OCR options
            col1, col2 = st.columns(2)
            with col1:
                ocr_lang = st.selectbox(
                    "OCR Language",
                    options=["eng", "fra", "deu", "spa", "ita", "por", "rus", "jpn", "kor", "chi_sim"],
                    index=0,
                    format_func=lambda x: {
                        "eng": "English", "fra": "French", "deu": "German",
                        "spa": "Spanish", "ita": "Italian", "por": "Portuguese",
                        "rus": "Russian", "jpn": "Japanese", "kor": "Korean", "chi_sim": "Chinese (Simplified)"
                    }.get(x, x)
                )
            with col2:
                preprocessing = st.selectbox(
                    "Image Preprocessing",
                    options=["None", "Threshold", "Blur", "Adaptive Threshold"],
                    index=0,
                    format_func=lambda x: x if x != "None" else "No Preprocessing"
                )

                # Map preprocessing options to function parameters
                preprocess_map = {
                    "None": None,
                    "Threshold": "thresh",
                    "Blur": "blur",
                    "Adaptive Threshold": "adaptive"
                }
                preprocess_param = preprocess_map[preprocessing]

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Create a container for each image
            image_container = st.container()
            with image_container:
                st.markdown(f"### Processing: {uploaded_file.name}")

                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Display the image
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(tmp_path, caption=uploaded_file.name)
                st.markdown("</div>", unsafe_allow_html=True)

                # Extract text button for this specific image
                if st.button(f"Extract Text", key=f"extract_text_{uploaded_file.name}"):
                    with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                        if extraction_method == "AI-based (GPT-4 Vision)":
                            # Use GPT-4 Vision for extraction
                            extraction_prompt = "Please extract all text visible in this image. Format it properly maintaining paragraphs, bullet points, and tables if present."
                            extraction_result = analyze_image_with_openai(tmp_path, extraction_prompt)
                            method_used = "AI-based extraction (GPT-4 Vision)"
                        else:
                            # Use OCR for extraction
                            if OCR_SUPPORT:
                                extraction_result = perform_ocr(tmp_path, lang=ocr_lang, preprocess=preprocess_param)
                                method_used = f"OCR (Tesseract) with {preprocessing} preprocessing, language: {ocr_lang}"
                            else:
                                extraction_result = "OCR is not available. Please install the required libraries."
                                method_used = "OCR (failed - libraries not available)"

                        if extraction_result:
                            st.markdown("<div class='analysis-result'>", unsafe_allow_html=True)
                            st.markdown("### Extracted Text")
                            st.markdown(f"*Method: {method_used}*")
                            st.markdown(extraction_result)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Save results
                            result_file = save_analysis_results(
                                uploaded_file.name,
                                f"Text extraction using {method_used}",
                                extraction_result,
                                tmp_path
                            )

                            # Add to chat history with user ID
                            user = get_current_user()
                            chat_entry = {
                                "image_name": uploaded_file.name,
                                "prompt": f"Text extraction using {method_used}",
                                "analysis": extraction_result,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "type": "text_extraction",
                                "user_id": user["user_id"] if user else None
                            }
                            st.session_state.image_chat_history.append(chat_entry)
                            save_image_chat_history(st.session_state.image_chat_history)

                            # Add to current session images
                            if uploaded_file.name not in [img["name"] for img in st.session_state.current_images]:
                                st.session_state.current_images.append({
                                    "name": uploaded_file.name,
                                    "path": tmp_path,
                                    "type": "text_extraction"
                                })

                            show_status(f"Extraction saved to {result_file} and added to history", "success")

                            # Download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                try:
                                    with open(result_file, "r") as f:
                                        st.download_button(
                                            label="📥 Download as JSON",
                                            data=f.read(),
                                            file_name=f"text_extraction_{uploaded_file.name.split('.')[0]}.json",
                                            mime="application/json",
                                            key=f"download_json_{uploaded_file.name}"
                                        )
                                except Exception as download_error:
                                    st.error(f"Error creating JSON download button: {download_error}")

                            with col2:
                                try:
                                    st.download_button(
                                        label="📥 Download as Text",
                                        data=extraction_result,
                                        file_name=f"text_extraction_{uploaded_file.name.split('.')[0]}.txt",
                                        mime="text/plain",
                                        key=f"download_text_{uploaded_file.name}"
                                    )
                                except Exception as download_error:
                                    st.error(f"Error creating text download button: {download_error}")

with tab3:
    st.header("Object Detection")

    # File uploader with multiple file support
    uploaded_files = st.file_uploader("Upload images for object detection", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="object_detection")

    if uploaded_files:
        # Detection options that apply to all images
        detection_type = st.selectbox(
            "Detection Type",
            options=["General Objects", "People and Faces", "Text and Signs", "Products and Logos", "Custom"],
            index=0
        )

        custom_prompt = ""
        if detection_type == "Custom":
            custom_prompt = st.text_area(
                "Custom Detection Prompt",
                value="Please identify and list all objects in this image with their approximate locations.",
                height=100
            )

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Create a container for each image
            image_container = st.container()
            with image_container:
                st.markdown(f"### Processing: {uploaded_file.name}")

                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Display the image
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(tmp_path, caption=uploaded_file.name)
                st.markdown("</div>", unsafe_allow_html=True)

                # Detect objects button for this specific image
                if st.button(f"Detect Objects", key=f"detect_objects_{uploaded_file.name}"):
                    with st.spinner(f"Detecting objects in {uploaded_file.name}..."):
                        # Set prompt based on detection type
                        if detection_type == "General Objects":
                            detection_prompt = "Please identify and list all objects in this image. For each object, provide: 1) Object name, 2) Approximate location in the image, 3) Brief description."
                        elif detection_type == "People and Faces":
                            detection_prompt = "Please identify all people in this image. For each person, describe: 1) Position in the image, 2) Approximate age group, 3) What they're wearing, 4) What they're doing. Do NOT include names or specific identities."
                        elif detection_type == "Text and Signs":
                            detection_prompt = "Please identify all text, signs, and written content in this image. For each text element, provide: 1) The text content, 2) Location in the image, 3) Type (sign, label, etc.)."
                        elif detection_type == "Products and Logos":
                            detection_prompt = "Please identify all products and logos in this image. For each item, provide: 1) Brand/product name if identifiable, 2) Location in the image, 3) Brief description."
                        else:
                            detection_prompt = custom_prompt

                        detection_result = analyze_image_with_openai(tmp_path, detection_prompt)

                        if detection_result:
                            st.markdown("<div class='analysis-result'>", unsafe_allow_html=True)
                            st.markdown("### Detection Results")
                            st.markdown(detection_result)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Save results
                            result_file = save_analysis_results(uploaded_file.name, detection_prompt, detection_result, tmp_path)

                            # Add to chat history with user ID
                            user = get_current_user()
                            chat_entry = {
                                "image_name": uploaded_file.name,
                                "prompt": detection_prompt,
                                "analysis": detection_result,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "type": "object_detection",
                                "user_id": user["user_id"] if user else None
                            }
                            st.session_state.image_chat_history.append(chat_entry)
                            save_image_chat_history(st.session_state.image_chat_history)

                            # Add to current session images
                            if uploaded_file.name not in [img["name"] for img in st.session_state.current_images]:
                                st.session_state.current_images.append({
                                    "name": uploaded_file.name,
                                    "path": tmp_path,
                                    "type": "object_detection"
                                })

                            show_status(f"Detection results saved to {result_file} and added to history", "success")

                            # Download button
                            try:
                                with open(result_file, "r") as f:
                                    st.download_button(
                                        label="📥 Download Detection Results",
                                        data=f.read(),
                                        file_name=f"object_detection_{uploaded_file.name.split('.')[0]}.json",
                                        mime="application/json",
                                        key=f"download_detection_{uploaded_file.name}"
                                    )
                            except Exception as download_error:
                                st.error(f"Error creating download button: {download_error}")

# Footer
st.markdown("---")
st.markdown("### 📊 Current Session Images")

# Display current session images
if st.session_state.current_images:
    st.markdown(f"**Images in current session:** {len(st.session_state.current_images)}")

    # Create columns for displaying images
    cols = st.columns(3)

    for i, img_data in enumerate(st.session_state.current_images):
        col_idx = i % 3
        with cols[col_idx]:
            st.markdown(f"**{img_data['name']}**")
            st.image(img_data['path'], width=150)
            st.markdown(f"*Type: {img_data['type']}*")

    # Add a button to clear session data
    if st.button("Clear Session Data", key="clear_image_session"):
        st.session_state.current_images = []
        st.session_state.image_session_id = str(uuid.uuid4())
        show_status("Session data cleared. You can upload new images.", "success")
        st.rerun()
else:
    st.info("No images processed in the current session.")

st.markdown("---")
st.markdown("### 📊 Analysis History")

# Add a button to clear analysis history
if st.button("Clear Analysis History", key="clear_analysis_history"):
    # Remove all analysis result files
    if os.path.exists("analysis_results"):
        try:
            analysis_files = [f for f in os.listdir("analysis_results") if f.endswith(".json")]
            for file in analysis_files:
                try:
                    os.remove(os.path.join("analysis_results", file))
                except Exception as e:
                    st.error(f"Error removing file {file}: {e}")
            show_status("Analysis history cleared successfully!", "success")
        except Exception as e:
            st.error(f"Error clearing analysis history: {e}")
    else:
        st.info("No analysis history to clear.")
    st.rerun()

# Create analysis_results directory if it doesn't exist
if not os.path.exists("analysis_results"):
    try:
        os.makedirs("analysis_results")
        show_status("Created analysis_results directory", "info")
    except Exception as e:
        st.error(f"Error creating analysis_results directory: {e}")

# Display previous analyses if available
if os.path.exists("analysis_results"):
    try:
        analysis_files = [f for f in os.listdir("analysis_results") if f.endswith(".json")]
    except Exception as e:
        st.error(f"Error reading analysis_results directory: {e}")
        analysis_files = []

    if analysis_files:
        analysis_files.sort(reverse=True)  # Most recent first

        for i, file in enumerate(analysis_files[:5]):  # Show only the 5 most recent
            try:
                with open(os.path.join("analysis_results", file), "r") as f:
                    try:
                        data = json.load(f)
                        with st.expander(f"{data.get('image_name', 'Unknown')} - {data.get('timestamp', 'Unknown date')}"):
                            st.markdown(f"**Prompt:** {data.get('prompt', 'No prompt')}")
                            st.markdown(f"**Analysis:** {data.get('analysis', 'No analysis')}")
                    except json.JSONDecodeError as json_error:
                        st.error(f"Error parsing JSON in file {file}: {json_error}")
                    except Exception as e:
                        st.error(f"Error processing analysis file {file}: {e}")
            except FileNotFoundError:
                st.warning(f"File {file} not found. It may have been moved or deleted.")
            except Exception as e:
                st.error(f"Error opening analysis file {file}: {e}")
    else:
        st.info("No previous analyses found.")
else:
    st.info("No analysis history available yet.")
