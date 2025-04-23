from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence

# 1. Embed & Vector Store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# 2. LLM with MCP
llm = ChatGroq(
    model="deepseek-coder:7b-instruct",
    temperature=0,
    streaming=True,
    enable_mcp=True  # this is the MCP flag
)

# 3. Prompt Template (Optional if you want MCP chunk alignment)
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:
{context}

Question: {question}
""")

# 4. LangChain Runnable with MCP-compatible flow
rag_chain = RunnableSequence(
    steps=[
        {"context": retriever, "question": lambda x: x["question"]},
        prompt,
        llm
    ]
)

# 5. Run it
response = rag_chain.invoke({"question": "What is LangChain used for?"})
print(response.content)
