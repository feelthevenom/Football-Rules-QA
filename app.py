from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import gradio as gr

# Load the PDF
loader = PDFPlumberLoader("Laws-of-the-Game-2024_25.pdf")
docs = loader.load()

# Split into chunks
text_splitter = SemanticChunker(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
documents = text_splitter.split_documents(docs)

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create the vector store and fill it with embeddings
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define llm
llm = OllamaLLM(model="llama3.2")

# Define the prompt template
prompt = PromptTemplate.from_template("""
1. Use the following pieces of context to answer the question at the end.
2. If you know about football rules and regulations, you can answer the question with your knowledge.
3. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.
4. Keep the answer crisp and limited to 3,4 sentences.
5. Answer the question related to foodball.

Context: {context}

Question: {question}

Helpful Answer:""")

# Create the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

def respond(question, history):
    return qa.invoke({"query": question})["result"]

# Create and launch the Gradio interface
gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me question related to football rules and regulations", container=False, scale=7),
    title="Football Chatbot",
    # examples=["What are different kinds of plant diseases", "What is Stewart's wilt disease"],
    cache_examples=True,
).launch(share=True)