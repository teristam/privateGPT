from nicegui import ui
from ZoteroLoader import ZoteroLoader, process_zotero_documents
import os 
import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from constants import CHROMA_SETTINGS
from ingest import *
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streaming_aiter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

dotenv.load_dotenv()

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
library_id = os.environ['ZOTERO_USER_ID']
api_key = os.environ['ZOTERO_API_KEY']
pdf_root_path = os.environ['PDF_BASE_PATH']
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
n_gpu_layer = os.environ.get('N_GPU_LAYERS')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
chunk_size = 500
chunk_overlap = 50

# start the zotero interface
zloader = ZoteroLoader(library_id, api_key, pdf_root_path)


class ZoteroOptions:
    def __init__(self) -> None:
        self.collection_name = 'Parkinsons'

options = ZoteroOptions()
global qa

def ingress():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={'device': 'cuda'})
    paths = zloader.get_file_path_collection(options.collection_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_zotero_documents(paths)
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_zotero_documents(paths)
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")

async def ask():
    text_message = text.value
    print(text_message)
    text.value = ''
    
    # Get the answer from the chain
    global qa
    res = qa(text_message)
    answer, docs = res['result'], res['source_documents']
    reply.value = answer

    # Print the result
    
    # print("\n\n> Question:")
    # print(query)
    # print("\n> Answer:")
    # print(answer)

    # # Print the relevant sources used for the answer
    # for document in docs:
    #     print("\n> " + document.metadata["source"] + ":")
    #     print(document.page_content)

def load_model():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    # callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM

    callbacks = [StreamingStdOutCallbackHandler()]

    llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks,n_gpu_layers=n_gpu_layer, verbose=False)

    global qa
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    print('Model loaded')

       
with ui.column().classes('w-full max-w-2xl mx-auto items-stretch'):
    with ui.row():
        ui.select(zloader.get_all_collections(), label='collection_select').bind_value(options, 'collection_name')
        ui.button(text='Ingress', on_click=ingress)

    with ui.row():
        ui.button(text='load model', on_click=load_model)
        text = ui.input(placeholder='message').props('rounded outlined input-class=mx-3') \
            .classes('w-full self-center').on('keydown.enter', ask)
            
        reply = ui.textarea().classes('w-full self-center')
    
ui.run()