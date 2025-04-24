# Document Knowledge Graph RAG

A Retrieval-Augmented Generation (RAG) application that creates knowledge graphs from documents and answers questions using both graph relationships and document content.

## Features

- **Document Processing**: Upload and process PDF, DOCX, and image files
- **Knowledge Graph Creation**: Automatically extract entities and relationships from documents
- **Intelligent Chunking**: Split documents into semantic chunks for better context preservation
- **Query-Specific Visualization**: See knowledge graphs tailored to your specific questions
- **Hybrid RAG**: Combine graph-based and text-based retrieval for more accurate answers
- **Duplicate Detection**: Prevent processing the same document multiple times

- ## Requirements

- Python 3.8+
- Neo4j Database
- OpenAI API Key
- Streamlit
- PyTesseract (for image processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anant-lad/KnowledgeWeaver.git
   cd KnowledgeWeaver
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Neo4j:
   - [Download and install Neo4j](https://neo4j.com/download/)
   - Create a new database or use an existing one
   - Note your Neo4j URI, username, and password

4. Create a `.env` file in the project root with the following variables:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Upload documents:
   - Use the file uploader to select PDF, DOCX, or image files
   - The application will process the documents and create a knowledge graph

3. Ask questions:
   - Type your question in the text input field
   - The application will search the knowledge graph and generate an answer
   - View the query-specific knowledge graph to see relationships related to your question

4. Explore the knowledge graph:
   - The full knowledge graph shows all relationships in the document
   - You can download the graph as an HTML file for offline viewing

## How It Works

1. **Document Processing**:
   - Documents are split into semantic chunks
   - Each chunk is processed to extract entities and relationships
   - Entities and relationships are stored in a Neo4j graph database

2. **Question Answering**:
   - Keywords are extracted from the question
   - A Cypher query is generated to find relevant entities and relationships
   - Text chunks containing those entities are retrieved
   - The answer is generated using both the graph structure and text content

3. **Knowledge Graph Visualization**:
   - Query-specific graph shows relationships related to the question
   - Full knowledge graph shows all relationships in the document

## Advanced Settings

- **Chunk Size**: Adjust the size of document chunks (default: 2000 characters)
- **Chunk Overlap**: Set the overlap between chunks for better context (default: 200 characters)
- **Debug Mode**: Enable to see additional processing information

## Limitations

- The quality of entity and relationship extraction depends on the OpenAI model
- Processing large documents may take time and consume API tokens
- The application works best with well-structured documents

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the web interface
- [Neo4j](https://neo4j.com/) for the graph database
- [OpenAI](https://openai.com/) for the language model
- [LangChain](https://langchain.readthedocs.io/) for the chat model integration
- [PyVis](https://pyvis.readthedocs.io/) for graph visualization
