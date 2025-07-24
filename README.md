# Content Generation Tool

A Streamlit-based web application for text processing, offering four powerful tools: Smart Summarization, Content Classification, AI-Assisted Story Generation, and Keyword Optimization. Built with Python, Transformers, NLTK, and scikit-learn, this tool leverages state-of-the-art NLP models to process and generate text efficiently.

## Features

- **Smart Summarization**: Summarizes long text into concise summaries using the T5-small model.
- **Content Classification**: Categorizes text into predefined categories (Health, Dailylife, Academics, Entertainment, Science) using zero-shot classification with BART-large-mnli.
- **AI-Assisted Story Generation**: Generates creative stories based on a user-specified topic and tone (e.g., Happy, Sad, Romantic) using DistilGPT-2.
- **Keyword Optimization**: Extracts the most relevant keywords from text using TF-IDF.

## Prerequisites

- **Python**: Version 3.8 or higher
- **Virtual Environment**: Recommended for dependency management
- **Internet Connection**: Required for downloading NLTK resources and Hugging Face models on first run
- **GPU**: Optional, for faster model inference (automatically detected if available)

## Project Structure

```
content-generation-tool/
├── Content Generation Tool.py  # Main application script
├── requirements.txt            # List of dependencies
├── README.md                  # Project documentation
```

## How to Run the Project

1. **Ensure Dependencies Are Installed**:
   Verify that all dependencies from `requirements.txt` are installed in your active Python environment.

2. **Run the Streamlit Application**:
   In your terminal, navigate to the project directory and execute:
   ```bash
   streamlit run "Content Generation Tool.py"
   ```
   Replace `"Content Generation Tool.py"` with the full path to the file if you’re not in the project directory, e.g.:
   ```bash
   streamlit run "/path/to/content-generation-tool/Content Generation Tool.py"
   ```

3. **Access the Web Interface**:
   - Streamlit will start a local server, typically at `http://localhost:8501`.
   - Open your web browser and navigate to `http://localhost:8501` to access the application.

## Usage

1. **Select a Tool**:
   - Choose from Smart Summarization, Content Classification, AI-Assisted Story Generation, or Keyword Optimization via the web interface.

2. **Provide Input**:
   - **Smart Summarization**: Enter text and specify maximum and minimum summary lengths (e.g., 50 and 20 words).
   - **Content Classification**: Enter text to categorize into Health, Dailylife, Academics, Entertainment, or Science.
   - **AI-Assisted Story Generation**: Provide a topic (e.g., "space adventure") and select a tone (e.g., Happy, Sad, Romantic, Action, Horror).
   - **Keyword Optimization**: Enter text to extract the top 3 relevant keywords.

3. **Submit**:
   - Click the relevant button (e.g., "Summarize", "Classify", "Generate Story", or "Extract Keywords") to process your input and view results.

4. **Example Usage**:
   - **Summarization**: Input a long article and set max length to 50 words to get a concise summary.
   - **Story Generation**: Enter "a magical forest" as the topic with a "Romantic" tone to generate a story.
   - **Classification**: Input a news article to classify it into categories like Health or Entertainment.
   - **Keyword Optimization**: Input a blog post to extract the top 3 keywords.
