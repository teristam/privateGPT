from nicegui import ui
from ZoteroLoader import ZoteroLoader, process_zotero_documents
import os 
import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from constants import CHROMA_SETTINGS
from ingest import *

dotenv.load_dotenv()

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50
library_id = os.environ['ZOTERO_USER_ID']
api_key = os.environ['ZOTERO_API_KEY']
pdf_root_path = os.environ['PDF_BASE_PATH']


# start the zotero interface
zloader = ZoteroLoader(library_id, api_key, pdf_root_path)


class ZoteroOptions:
    def __init__(self) -> None:
        self.collection_name = 'Parkinsons'

options = ZoteroOptions()

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
    

with ui.row():
    ui.select(zloader.get_all_collections(), label='collection_select').bind_value(options, 'collection_name')
    ui.button(text='Ingress', on_click=ingress)
    
    
ui.run()