import streamlit as st
import nltk
from transformers import pipeline, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk.data
import torch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Function to download NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource.startswith('punkt') else f'corpora/{resource}')
            print(f"NLTK resource '{resource}' already downloaded.")
        except LookupError:
            print(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource)

# Download NLTK resources before starting the app
download_nltk_resources()

# Cached pipeline for summarization
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small", device=0 if torch.cuda.is_available() else -1)

# Cached pipeline for classification
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Cached pipeline for story generation
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")

# Section 1: Tool Descriptions
def display_tool_descriptions():
    return [
        ("Smart Summarization", "Summarizes your text into a concise summary."),
        ("Content Classification",
         "Categorizes your text into Health, Dailylife, Academics, Entertainment, or Science."),
        ("AI-Assisted Story Generation", "Generates a creative story around a topic and tone."),
        ("Keyword Optimization", "Extracts top relevant keywords.")
    ]

# Section 2: Smart Summarization
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    filtered_sentences = [sent for sent in sentences if len([w for w in word_tokenize(sent) if w.lower() not in stop_words]) > 3]
    return " ".join(filtered_sentences[:5])

def summarize_text(text, max_length=50, min_length=20):
    try:
        summarizer = load_summarizer()
        input_text = preprocess_text(text)
        input_text = "Summarize the following text in a concise manner: " + input_text
        with st.spinner("Summarizing text..."):
            summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
        summary_text = summary[0]['summary_text']
        st.subheader("Summarization Output")
        st.write(f"**Summary:** {summary_text}")
        return summary_text
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return None

# Section 3: Content Classification
def classify_text(text, categories=["Health", "Dailylife", "Academics", "Entertainment", "Science"]):
    try:
        classifier = load_classifier()
        result = classifier(text, candidate_labels=categories)
        label = result['labels'][0]
        score = result['scores'][0]
        st.subheader("Classification Output")
        st.write(f"**Category:** {label}, **Score:** {score:.4f}")
        return {"label": label, "score": score}
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        return None

# Section 4: Story Telling
def generate_story(topic, tone):
    try:
        generator = load_generator()
        prompt = f"{tone.capitalize()} story about {topic}:\n\nOnce upon a time,"
        result = generator(
            prompt,
            max_length=70,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            no_repeat_ngram_size=2
        )[0]["generated_text"]
        sentences = nltk.sent_tokenize(result)
        story = ' '.join(sentences).strip()
        st.subheader("Generated Story")
        st.write(f"**Tone:** {tone}, **Topic:** {topic}")
        st.write(story)
        return story
    except Exception as e:
        st.error(f"Error generating story: {str(e)}")
        return None

# Section 5: Keyword Optimization
def extract_keywords(text):
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        word_scores = [(feature_names[i], tfidf_scores[i]) for i in range(len(feature_names))]
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
        keywords = [word for word, score in word_scores[:3]]
        st.subheader("Keyword Optimization Output")
        st.write(f"**Most Relevant Keywords:** {keywords}")
        return keywords
    except Exception as e:
        st.error(f"Error in keyword optimization: {str(e)}")
        return None

# Streamlit UI
def main():
    st.title("Text Processing Tools")
    st.write("Select a tool below to process your text or generate a story.")

    # Display tools
    tools = display_tool_descriptions()
    st.subheader("Available Tools")
    selected_tool = st.session_state.get('selected_tool', None)

    # Tool selection buttons
    for tool_name, description in tools:
        if st.button(f"{tool_name}", key=tool_name):
            st.session_state.selected_tool = tool_name
        st.write(f"**{tool_name}:** {description}")

    # Handle selected tool
    if st.session_state.get('selected_tool'):
        selected_tool = st.session_state.selected_tool
        st.subheader(f"Using {selected_tool}")

        # Input for tools requiring text
        if selected_tool in ["Smart Summarization", "Content Classification", "Keyword Optimization"]:
            with st.form(key=f"{selected_tool}_form"):
                user_text = st.text_area("Enter text to process:", height=200, key=f"{selected_tool}_text")
                submit_button = None

                # Tool-specific inputs and execution
                if selected_tool == "Smart Summarization":
                    max_length = st.number_input("Max summary length:", min_value=10, step=10, value=50,
                                                 key="summarization_max")
                    min_length = st.number_input("Min summary length:", min_value=10, step=10, value=20,
                                                 key="summarization_min")
                    submit_button = st.form_submit_button("Summarize")
                    if submit_button:
                        if user_text:
                            summarize_text(user_text, max_length, min_length)
                        else:
                            st.warning("Please provide text to summarize.")

                elif selected_tool == "Content Classification":
                    submit_button = st.form_submit_button("Classify")
                    if submit_button:
                        if user_text:
                            classify_text(user_text)
                        else:
                            st.warning("Please provide text to classify.")

                elif selected_tool == "Keyword Optimization":
                    submit_button = st.form_submit_button("Extract Keywords")
                    if submit_button:
                        if user_text:
                            extract_keywords(user_text)
                        else:
                            st.warning("Please provide text to extract keywords.")

        elif selected_tool == "AI-Assisted Story Generation":
            with st.form(key="story_generation_form"):
                story_topic = st.text_input("Enter a story topic:", key="story_topic")
                story_tone = st.selectbox("Select tone:", ["", "Happy", "Sad", "Romantic", "Action", "Horror"],
                                          key="story_tone")
                submit_button = st.form_submit_button("Generate Story")
                if submit_button:
                    if story_topic and story_tone:
                        generate_story(story_topic, story_tone)
                    else:
                        st.warning("Please provide both topic and tone.")

if __name__ == "__main__":
    main()

# To run the application, use in the terminal:
# streamlit run "path/to/Content Generation Tool.py"