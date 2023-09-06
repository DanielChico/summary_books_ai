import ai21
from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from server.core.llm_utils import set_OpenAI_llm_model
from server.core.prompt_utils import get_prompt_for_translation

def answer():
    """
    Translates a text to a given language. The translation is made by a LLM.
    :param text: The text to translate.
    :param language: The language to translate the text to.
    :return: The text translated by the LLM.
    """
    # Create LLMChain with custom prompt for translation
    llm = set_OpenAI_llm_model(text)
    prompt = get_prompt_predict()
    chain = LLMChain(llm=llm, prompt=prompt)

    # Return the translated text by running the prompt with the given input variables
    return chain.predict(history=text, language=language)

def get_prompt_predict() -> PromptTemplate:
    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Documents relevant to the conversation:
    {history}
    Document summaries:
    {summaries}
    Human: {input}
    AI:
    """
    input_variables = ['history', 'input', 'summaries']

    return PromptTemplate(
        input_variables=input_variables,
        template=template
    )