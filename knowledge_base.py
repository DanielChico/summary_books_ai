import faiss
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from pathlib import Path


def get_vector_store(file_path: str) -> FAISS:
    """
    Retrieves or creates a FAISS-based vector store using the given file path.

    :param file_path: The path to the file where the vector store is saved or to be saved.
    :return: The FAISS-based vector store.
    """
    vectorstore: FAISS

    # If the vector store file doesn't exist, create a new one
    if not Path(file_path).exists():
        # Create FAISS-based vector store
        embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
        index = faiss.IndexFlatL2(embedding_size)
        embedding_function = OpenAIEmbeddings().embed_query
        vectorstore = FAISS(embedding_function, index, InMemoryDocstore({}), {})

        # Save a default context in the memory of the vector store
        retriever = vectorstore.as_retriever()
        memory = VectorStoreRetrieverMemory(retriever=retriever)
        memory.save_context({"input": "."}, {"output": "."})

        # Save the created vector store to the given file path
        vectorstore.save_local(file_path)
    # If the vector store exists, load it from the file
    else:
        vectorstore = FAISS.load_local(file_path, OpenAIEmbeddings())

    return vectorstore
    

def get_text_segments(source: str, sourceType: str) -> list[str]:
    """
    Splits text into segments using the AI21 Studio API.

    :param source: The text to split into segments.
    :param sourceType: The type of the source. Must be "TEXT" or "URL".
    :return: A list of text segments.
    """
    # Initialize an empty list to store the segments
    segments = []

    # Initialize a list containing the source text
    texts: list[str] = [source]

    # Initialize a counter variable
    i = 0

    # If the source type is "TEXT" and there are still texts to process
    while sourceType == "TEXT" and i < len(texts):
        # If the current text is longer than 100000 characters
        if len(texts[i]) > 100000:
            # Split the text in half with an overlap of 100 characters
            middle_index = int(len(texts[i]) / 2)
            first_half = texts[i][:middle_index + 100]
            second_half = texts[i][middle_index - 100:]

            # Remove the current text from the list and insert its two halves
            index = texts.index(texts[i])
            texts.remove(texts[i])
            texts.insert(index, second_half)
            texts.insert(index, first_half)
        else:
            # Move on to the next text
            i += 1

    # Iterate over the texts
    for text in texts:
        # Use the AI21 Studio API to split the text into segments
        response = ai21.Segmentation.execute(
            source=text,
            sourceType=sourceType
        )

        # Extract the segments from the response
        segments_list = response["segments"]

        # Append each segment to the segments list
        for dict in segments_list:
            segments.append(dict['segmentText'])

    # Return the list of segments
    return segments