from pyzotero import zotero
import os


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
    

