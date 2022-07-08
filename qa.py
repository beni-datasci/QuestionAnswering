from transformers import pipeline
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_qa_model():
    model = pipeline("question-answering")
    return model

qa = load_qa_model()
st.title("Ask Questions about your Text")
sentence = st.text_area('Please paste your text :', height=30)
question = st.text_input("Your Questions regarding the text?")
button = st.button("Get me Answers")
with st.spinner("Discovering Answers.."):
    if button and sentence:
        answers = qa(question=question, context=sentence)
        st.write(answers['answer'])
