import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from config import groq_config

os.environ["GROQ_API_KEY"] = groq_config.groq_api_key

def get_llama_response(input_text, number_of_words, blog_style):
    # Using the ChatGroq class
    llm = ChatGroq(model_name="llama3-8b-8192", 
                    temperature=0.6, 
                    max_tokens=2 * number_of_words)

    prompt_template = PromptTemplate.from_template(
        "Write a complete blog post for {blog_style} on the topic of {input_text} with a maximum of {number_of_words} words."
    )

    # Convert prompt to string
    prompt = prompt_template.invoke(
        {
            "input_text": input_text,
            "number_of_words": number_of_words,
            "blog_style": blog_style
        }
    )
    print(prompt)

    ## Generate response from the Llama3.3 model

    # Without streaming
    # llm_response = llm.invoke(prompt) 
    # print(llm_response.content)
    # return llm_response.content

    # With streaming
    # return (token.content for token in llm.stream(prompt))
    for token in llm.stream(prompt):
        print(token.content, end='', flush=True)
        yield token.content
    print("\n")
    
st.set_page_config(page_title="Blog Generator", 
                    page_icon="📝", 
                    layout="centered", 
                    initial_sidebar_state="collapsed")

st.title("Generate Blog Posts with LLama3.3 📝")

input_text = st.text_input("Enter the topic:")
no_words = st.number_input("Enter the maximum number of words:", min_value=100, step=50)
blog_style = st.selectbox("Select the blog style:", ["Researcher", "DataScientists", "CommonPeople"])

if st.button("Generate Blog Post"):
    st.write(get_llama_response(input_text, no_words, blog_style))
