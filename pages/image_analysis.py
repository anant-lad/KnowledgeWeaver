import streamlit as st
import os
import tempfile
import base64
import json
from datetime import datetime
from dotenv import load_dotenv

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

# Initialize OpenAI client
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    st.error("OpenAI API key not found. Please add it to your .env file as OPENAI_API_KEY.")
    client = None

# Page configuration
st.set_page_config(page_title="Image Analysis - KnowledgeWeaver", layout="wide")

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

# Main page content
st.title("🖼️ Image Analysis")
st.markdown("Upload images for detailed AI-powered analysis")

# Create tabs for different analysis types
tab1, tab2, tab3 = st.tabs(["General Analysis", "Text Extraction", "Object Detection"])

with tab1:
    st.header("General Image Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="general_analysis")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Display the image
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(tmp_path, caption=uploaded_file.name)
        st.markdown("</div>", unsafe_allow_html=True)

        # Analysis options
        analysis_prompt = st.text_area(
            "Analysis Prompt",
            value="Please analyze this image in detail. Describe what you see, including objects, people, colors, and the overall scene.",
            height=100
        )

        # Analysis button
        if st.button("Analyze Image", key="analyze_general"):
            with st.spinner("Analyzing image..."):
                analysis_result = analyze_image_with_openai(tmp_path, analysis_prompt)

                if analysis_result:
                    st.markdown("<div class='analysis-result'>", unsafe_allow_html=True)
                    st.markdown("### Analysis Result")
                    st.markdown(analysis_result)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Save results
                    result_file = save_analysis_results(uploaded_file.name, analysis_prompt, analysis_result, tmp_path)
                    show_status(f"Analysis saved to {result_file}", "success")

                    # Download button
                    try:
                        with open(result_file, "r") as f:
                            st.download_button(
                                label="📥 Download Analysis",
                                data=f.read(),
                                file_name=f"analysis_{uploaded_file.name.split('.')[0]}.json",
                                mime="application/json"
                            )
                    except Exception as download_error:
                        st.error(f"Error creating download button: {download_error}")

with tab2:
    st.header("Text Extraction from Images")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image containing text", type=["jpg", "jpeg", "png"], key="text_extraction")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Display the image
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(tmp_path, caption=uploaded_file.name)
        st.markdown("</div>", unsafe_allow_html=True)

        # Extraction options
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

        # Analysis button
        if st.button("Extract Text", key="extract_text"):
            with st.spinner("Extracting text from image..."):
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
                    show_status(f"Extraction saved to {result_file}", "success")

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        try:
                            with open(result_file, "r") as f:
                                st.download_button(
                                    label="📥 Download as JSON",
                                    data=f.read(),
                                    file_name=f"text_extraction_{uploaded_file.name.split('.')[0]}.json",
                                    mime="application/json"
                                )
                        except Exception as download_error:
                            st.error(f"Error creating JSON download button: {download_error}")

                    with col2:
                        try:
                            st.download_button(
                                label="📥 Download as Text",
                                data=extraction_result,
                                file_name=f"text_extraction_{uploaded_file.name.split('.')[0]}.txt",
                                mime="text/plain"
                            )
                        except Exception as download_error:
                            st.error(f"Error creating text download button: {download_error}")

with tab3:
    st.header("Object Detection")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image for object detection", type=["jpg", "jpeg", "png"], key="object_detection")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Display the image
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(tmp_path, caption=uploaded_file.name)
        st.markdown("</div>", unsafe_allow_html=True)

        # Analysis options
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

        # Analysis button
        if st.button("Detect Objects", key="detect_objects"):
            with st.spinner("Detecting objects in image..."):
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
                    show_status(f"Detection results saved to {result_file}", "success")

                    # Download button
                    try:
                        with open(result_file, "r") as f:
                            st.download_button(
                                label="📥 Download Detection Results",
                                data=f.read(),
                                file_name=f"object_detection_{uploaded_file.name.split('.')[0]}.json",
                                mime="application/json"
                            )
                    except Exception as download_error:
                        st.error(f"Error creating download button: {download_error}")

# Footer
st.markdown("---")
st.markdown("### 📊 Analysis History")

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
