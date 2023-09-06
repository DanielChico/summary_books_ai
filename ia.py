from fastapi import APIRouter, Body, HTTPException
from ..core.knowledge_base import get_vector_store
from ..core.text_utils import get_text_segments
from ..core.file_utils import delete_directory
from ..core.memory import add_new_doc_to_chat_memory

router = APIRouter()


def text_doc(text: str):
    """
    Add a text document to the vector store for the given conversation ID.

    :param conversation_id: A string representing the ID of the conversation to add the text document to.
    :param text: A string representing the text document to add to the vector store. The default value is taken from the request body with media type "text/plain".
    :return: A dictionary containing a single key-value pair. The key is "message" and the value is a string representing that the process was successful.
    """
    # Search the vector store related to the conversation_id
    vector_store = get_vector_store(f"vector_stores")

    # Split text into meaningful segments
    segments = get_text_segments(text, "TEXT")

    # Add text segments to the vector store
    vector_store.add_texts(texts=segments)

    # Save updated vector store to local filesystem
    vector_store.save_local(f"vector_stores")

    # Save in chat memory that the document was added
    add_new_doc_to_chat_memory()

    return {"message": "OK"}