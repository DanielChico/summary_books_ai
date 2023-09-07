import ai21
from fpdf import FPDF
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader


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


def create_temporary_pdf(path, text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(path)

# create_temporary_pdf("archivo.pdf", "Este es un texto largo con la letra Ñ.")


def load_pdf(path: str):

    loader = PyPDFLoader(path)
    docs = loader.load_and_split()

    # print('pass')

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = load_summarize_chain(llm, chain_type="refine")

    return chain.run(docs)
