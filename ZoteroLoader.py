from multiprocessing import Pool
from typing import List
from pyzotero import zotero
import os
from langchain.docstore.document import Document
from tqdm import tqdm
from ingest import load_single_document
from langchain.text_splitter import RecursiveCharacterTextSplitter

chunk_size = 500
chunk_overlap = 50


def get_file_path(item):
    try:
        path = item['data']['path'][12:]
        return path
    except KeyError:
        return None    

class ZoteroLoader:
    def __init__(self, library_id, api_key, pdf_root_path):
        self.zot = zotero.Zotero(library_id, 'user', api_key)
        self.pdf_root_path = pdf_root_path
        
    def get_items_in_collection(self, collection_name, **kwargs):
        collection = self.zot.collections(q=collection_name, limit=1)
        collection_id =  collection[0]['key']
        items = self.zot.collection_items(collection_id, **kwargs)
        return items
    
        
    def get_file_path_collection(self, collection_name, **kwargs):
        items = self.get_items_in_collection(collection_name, **kwargs)
        file_names = [get_file_path(item) for item in items]
        
        return [os.path.join(self.pdf_root_path, fn) for fn in file_names if fn is not None]    
    
    def get_all_collections(self):
        collections = self.zot.collections()
        
        return [col['data']['name'] for col in collections] 
    
    
def load_zotero_documents(doc_paths:List[str]) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(doc_paths), desc='Loading new documents', ncols=80) as pbar:
            for i, doc in enumerate(pool.imap_unordered(load_single_document, doc_paths)):
                results.append(doc)
                pbar.update()

    return results

def process_zotero_documents(doc_paths) -> List[Document]:
    """
    Load documents and split in chunks
    """
    documents = load_zotero_documents(doc_paths)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts