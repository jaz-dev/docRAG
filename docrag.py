import argparse
import os
import openai, langchain, langchain_openai, pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader


INDEX = os.environ["INDEX"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
embed_model = "text-embedding-3-large"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model = embed_model)


def is_pdf(file_path):
    return file_path.lower().endswith('.pdf')

def add_file_to_index(file_path, namespace):
    if (is_pdf(file_path)):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        file_content = ''.join([page.page_content for page in pages])
        file_content = file_content.replace('\n', '').replace('\r', '')
    else:
        with open(file_path, 'r') as file_data:
            file_content = file_data.read().replace('\n', '').replace('\r', '')

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 0,
    length_function = len)
    doc_texts = text_splitter.create_documents([file_content])

    PineconeVectorStore.from_texts([t.page_content for t in doc_texts], embeddings, index_name=INDEX, namespace=namespace)

def query(query, namespace):
    doc_search = PineconeVectorStore.from_existing_index(index_name= INDEX, embedding=embeddings, namespace=namespace)
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4o")

    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:
    """
    retriever = doc_search.as_retriever()
    custom_rag_prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    rag_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=custom_rag_prompt,
        output_parser=StrOutputParser())
    context = retriever.invoke(query)
    response = rag_chain.invoke({
        "question": query,
        "context": context
    })
    print(response)

def main():
    parser = argparse.ArgumentParser(description='Process input file or question with a namespace.')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file_path', metavar='file_path', help='Specify the file location.')
    group.add_argument('-q', '--query', metavar='query', nargs='?', const=True, help='Specify the query.')

    parser.add_argument('-n', '--namespace', required=False, metavar='name_space', help='Specify the namespace.')

    args = parser.parse_args()

    if args.file_path:
        print(f"Indexing: {args.file_path}")
        print(f"Using namespace: {args.namespace}")

        add_file_to_index(args.file_path, args.namespace)
    elif args.query:
        if args.query is True:
            while True:
                user_input = input("query: ")
                if user_input.strip().lower() == 'exit()':
                    break
                else:
                    query(user_input, args.namespace)
        else:
            query(args.query, args.namespace)

if __name__ == '__main__':
    main()
    