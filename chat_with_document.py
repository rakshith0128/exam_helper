import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
import os

def load_document(file):
    import os
    name,extension = os.path.splitext(file)

    if extension ==  '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f"loading {file}")
        loader = PyPDFLoader(file)

    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f"loading {file}")
        loader = Docx2txtLoader(file)

    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print("Document format is not supported")
        return None

    data = loader.load()
    return data

# wikipedia
def load_from_wikipedia(query, lang = 'en',load_max_docs = 1):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang = lang, load_max_docs = load_max_docs)
    data = loader.load()
    return data

def chunk_data(data,chunk_size = 256, chunk_overlap = 20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
         model="models/embedding-001"
    )
    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store

def ask_and_get_answer(vector_store, q, k=3):
    # k is the number of chunks being shown to gemini 
    
    from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=1, convert_system_message_to_human=True)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

   
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Invoke the chain with the question.
    answer = chain.invoke(q)
    return answer


def calculate_embeddings_cost(texts):
    # Estimate: 1 token ≈ 4 characters
    total_chars = sum(len(page.page_content) for page in texts)
    estimated_tokens = total_chars // 4

    # Cost per 1K tokens (rough estimate; adjust if Google shares pricing)
    cost_per_1k_tokens = 0.0001
    estimated_cost = (estimated_tokens / 1000) * cost_per_1k_tokens

    # print(f"Estimated Tokens: {estimated_tokens}")
    # print(f"Estimated Cost (USD): ${estimated_cost:.6f}")
    return(estimated_tokens,estimated_cost)

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    from PIL import Image
    load_dotenv(find_dotenv(), override = True)

    img = Image.open("gemini_2.png")
    resized_img = img.resize((500, 150))  

    st.image(resized_img)

    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        api_key = st.text_input('Gemini API Key (optional) :', type='password')
        if api_key:
            os.environ['Gemini API Key'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type = ['pdf','docx','txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100,max_value=2048, value = 512, on_change=clear_history)

        k = st.number_input('k', min_value=1, max_value=20, value = 3, on_change=clear_history)
        add_data = st.button('Add', on_click =clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size = chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')


                tokens, embedding_cost = calculate_embeddings_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)

                st.session_state.vs = vector_store
                st.success("File uploaded, chunked and embedded succcesfully.")

    
    q = st.text_input('Ask a question about the content of your file:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs 
            # st.write(f'k:{k}')
            answer = ask_and_get_answer(vector_store, q,k)
            st.text_area('LLM answer: ', value=answer['result'],height = 100)
    
            st.divider()

            if'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer["result"]}'
            st.session_state.history =f'{value} \n {"-"*130} \n {st.session_state.history}'
            h = st.session_state.history

            st.text_area(label='Chat History',  key ='history', height = 400)