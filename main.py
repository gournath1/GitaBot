from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'
function_completed = False
function_running = False

def load():
    import os
    os.environ["OPENAI_API_KEY"] = "OPEN_AI_KEY"
    from langchain.document_loaders import PyPDFLoader
    from langchain.indexes import VectorstoreIndexCreator
    from langchain import OpenAI
    from langchain.docstore.document import Document
    from langchain.chains.question_answering import load_qa_chain
    from langchain.indexes.vectorstore import VectorstoreIndexCreator
    from langchain.document_loaders import TextLoader
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.prompts import PromptTemplate
    loader = PyPDFLoader("sgita.pdf")
    #loader=TextLoader("gita.txt")
    index = VectorstoreIndexCreator().from_loaders([loader])
    embeddings = OpenAIEmbeddings()
    index_creator = VectorstoreIndexCreator()
    global docsearch
    docsearch = index_creator.from_loaders([loader])
    global llm
    llm = OpenAI(temperature=0.2)
    global chain
    chain = load_qa_chain(llm, chain_type="stuff")
    global function_completed
    function_completed = True
    print('Load Complete ',function_completed)
    
def chatbot_response(query):
    global function_completed
    response="Please wait while loading"
    if function_completed:
        docs = docsearch.vectorstore.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
    return response

def start_load():
    global function_completed
    global function_running
    print('Load Complete ',function_completed ,' Load Running ', function_running)
    if not function_running:
        # Start the long-running function in a separate thread
        function_running = True
        load()

@app.route("/")
def home():
    start_load()
    global function_completed
    if not function_completed:
        return render_template('loader.html')
    else:
        return render_template('index.html') 
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == '__main__':
    app.run()