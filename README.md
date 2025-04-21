# KnowledgeWeaver

KnowledgeWeaver is an advanced document management application with AI-powered search and analysis capabilities. It allows users to upload, process, and query multiple document formats while leveraging vector embeddings and large language models.

## Features

- **Multi-format Document Support**: Upload and process PDF, DOCX, JPG, and PNG files
- **Advanced Search**: Ask natural language questions about your documents
- **Vector Embeddings**: Uses OpenAI embedding models to create semantic representations of documents
- **Neo4j Vector Database**: Stores document embeddings for efficient retrieval
- **Image Analysis**: Dedicated page for analyzing images with AI
  - General image analysis
  - Text extraction (OCR and AI-based)
  - Object detection
- **Filtering Options**: Filter documents by file type, name, and upload date
- **Chat History**: Track and review previous queries and answers
- **Document Comparison**: Compare information across different source documents
- **Export Functionality**: Download answers and context as JSON

## Requirements

- Python 3.7+
- Neo4j Database
- OpenAI API Key
- Tesseract OCR (optional, for enhanced OCR capabilities)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd knowledgeweaver
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. For OCR support (optional):
   ```
   pip install pytesseract opencv-python numpy
   ```
   You'll also need to install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on your system.

4. Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   NEO4J_URL=your_neo4j_url
   NEO4J_USER=your_neo4j_username
   NEO4J_PASSWORD=your_neo4j_password
   ```

## Usage

1. Start the application:
   ```
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Use the interface to:
   - Upload documents
   - Select embedding models
   - Ask questions about your documents
   - Filter results
   - Analyze images

## Main Components

### Document Management (Main Page)
- Upload documents (PDF, DOCX, JPG, PNG)
- Select embedding models (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
- Process documents into vector embeddings
- Query documents with natural language
- Filter results by file type, name, and upload date

### Image Analysis (Image Analysis Page)
- General image analysis with AI
- Text extraction using OCR or AI-based methods
- Object detection with customizable detection types

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `NEO4J_URL`: URL for your Neo4j database
- `NEO4J_USER`: Username for Neo4j database
- `NEO4J_PASSWORD`: Password for Neo4j database

## Embedding Models

The application supports the following OpenAI embedding models:
- **text-embedding-3-small**: Efficient text embedding model with 1536 dimensions
- **text-embedding-3-large**: High-performance text embedding model with 3072 dimensions
- **text-embedding-ada-002**: Legacy text embedding model with 1536 dimensions

## Troubleshooting

- If you encounter Neo4j vector index errors, the application will attempt to recreate the index
- For OCR functionality, ensure Tesseract is properly installed on your system
- Check the console for detailed error messages

## License

[Your License Information]

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/)
- Uses [LangChain](https://www.langchain.com/) for document processing and retrieval
- Powered by [OpenAI](https://openai.com/) models for embeddings and analysis
- [Neo4j](https://neo4j.com/) for vector database storage
