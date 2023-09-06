from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from knowledge_base import get_vector_store


def main():
    load_dotenv()
    st.set_page_config(page_title='Ask your PDF')
    st.header("Ask your PDF 📄")

    pdfs = st.file_uploader('Upload your PDF', type='pdf', accept_multiple_files=True)


    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)

        # split text
        text_spliter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_spliter.split_text(text)

        # vector_store = get_vector_store('/vector_store')
        # vector_store.add_texts(texts=chunks)

        # create embeddings
        # error api_key
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        chain = load_summarize_chain(llm, chain_type="stuff")
        summarize = chain.run(question="Resume")

    
    if pdfs:
        # show user input
        try:
            user_question = st.text_input('Ask a question about your PDF:')
        except Exception:
            print('error')

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                docs = knowledge_base.similarity_search(user_question)
                chain = load_qa_chain(OpenAI(), chain_type='stuff')
                response = chain.run(input_documents=docs, question=user_question)

                st.write(response)


if __name__ == '__main__':
    main()
