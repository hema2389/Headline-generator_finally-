import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# App Title and Description
st.title("Headline Generator")
st.write("Generate headlines from news articles using a pre-trained BART model!")

# Load Model and Tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/bart-base"  # Replace with your GitHub model path if custom
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Function to Generate Headline
def generate_headline(article, max_length=64, num_beams=5):
    inputs = tokenizer(
        article, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
    )
    headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return headline

# User Input
st.header("Input Your News Article")
article_text = st.text_area("Paste the article content below:")

if st.button("Generate Headline"):
    if article_text.strip():
        with st.spinner("Generating headline..."):
            headline = generate_headline(article_text)
        st.success("Generated Headline:")
        st.write(headline)
    else:
        st.warning("Please provide the article text.")

# Footer
st.write("---")
st.markdown("Created with ❤️ using [Streamlit](https://streamlit.io/) and [Hugging Face Transformers](https://huggingface.co/).")
