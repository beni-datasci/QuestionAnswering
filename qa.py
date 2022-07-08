from transformers import pipeline
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_qa_model():
    model = pipeline("question-answering")
    return model
st.title("The Question Answering-Machine")
st.write(" ")
st.write("Our beginnings")
with st.expander("Our Team & Mission"):
  st.write("Hello! Our team consists of Frenki Pushaj & Benjamin Rattanpal.")
  st.write("Our mission consists of creating an app that answers all your questions regarding celebrities & even more.")
  st.write("We strive to achieve this by utilizing Machine Learning :)")
  
with st.expander("Which Dataset do we use?"):
  st.write("We use the Squad QA dataset provided by rajpurkar on huggingface.co . It`s regarded as one of the most popular Datasets for Question Answering.")
 
st.write("Examples from our Dataset")

with st.expander("In what city and state did Beyonce grow up?"):
  st.write("Houston, Texas")
  st.write("ID: 56bf6b0f3aeaaa14008c9601")

with st.expander("When did Beyonce release Dangerously in Love?"):
  st.write("2003")
  st.write("ID: 56d43c5f2ccc5a1400d830ac")
    
qa = load_qa_model()
st.write("Create your own questions")
with st.expander("Curious?"):
    st.title("Ask Questions about your Text")
    sentence = st.text_area('Please paste your text :', height=30)
    question = st.text_input("Your Questions regarding the text?")
    button = st.button("Get me Answers")
    with st.spinner("Discovering Answers.."):
        if button and sentence:
            answers = qa(question=question, context=sentence)
            st.write(answers['answer'])
