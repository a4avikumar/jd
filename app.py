# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3, max_tokens=800)
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate

# # System prompt to allow external knowledge and past case references
# system_prompt = (
#     "You are a legal advisor specializing in Pakistan's criminal law. "
#     "Use both the provided text and your own legal knowledge to generate a comprehensive response. "
#     "Your response should include:\n"
#     "- Relevant legal sections and laws\n"
#     "- Citations \n"
#     "- Must give me References to past similar cases, their judgments, and outcomes\n"
#     "- Additional legal insights to provide a complete answer\n\n"
#     "If there is any precedent or past case related to the query, mention it along with its citation and outcome.\n\n"
#     "Context:\n{context}"
# )

# # Creating the chat prompt template
# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}"),
# ])
# vectorstore = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# )
# retriever = vectorstore.as_retriever()
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)


from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Load saved ChromaDB vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)

retriever = vectorstore.as_retriever()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=800)

# Define system prompt
system_prompt = (
    "You are a legal advisor specializing in Pakistan's criminal law. "
    "Use both the provided text and your own legal knowledge to generate a comprehensive response. "
    "Your response should include:\n"
    "- Relevant legal sections and laws\n"
    "- Citations \n"
    "- References to past similar cases, their judgments, and outcomes\n"
    "- Additional legal insights to provide a complete answer\n\n"
    "If there is any precedent or past case related to the query, mention it along with its citation and outcome.\n\n"
    "Context:\n{context}"
)

# Create chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/query", methods=["POST"])
def query():
    """
    Flask API endpoint to handle user queries.
    """
    data = request.json
    user_query = data.get("query", "")
    
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    # Generate response using RAG
    response = rag_chain.invoke({"input": user_query})
    
    return jsonify({"response": response["output_text"]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
