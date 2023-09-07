from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import ai21
import openai

from src.predict import answer 


def main():
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    ai21.api_key = os.getenv('AI21_API_KEY')
    st.set_page_config(page_title='Ask your PDF')
    st.header("Ask your PDF 📄")

    pdfs = st.file_uploader('Upload your PDF', type='pdf', accept_multiple_files=True)

    text = ''
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)
    
    if pdfs:
        # split text
        text_spliter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_spliter.split_text(text)


        # create embeddings
        # error api_key
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        # show user input
        try:
            user_question = st.text_input('Ask a question about your PDF:')
        except Exception:
            print('error')

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                docs = knowledge_base.similarity_search(user_question)
                print(docs)
                # chain = load_qa_chain(OpenAI(), chain_type='stuff')
                # response = chain.run(input_documents=docs, question=user_question)
                response = answer(context=docs, text= user_question, summaries='')

                st.write(response)


if __name__ == '__main__':
    main()
