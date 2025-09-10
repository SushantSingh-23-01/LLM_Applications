import os
import hashlib
import ollama
import chromadb
from chromadb.config import Settings
from typing import List, Optional
from text_splitters import sentence_aware_splitter
import time

class ParentChildIngester:
    """
        Performs Parent Child Ingestion of text.
        - Child Chunk -> Smaller in size -> Leads to higher semantic matching between query and chunk.
        - Parent Chunk -> Larger in size -> Leads to more Context for LLM to provide appropriate answer.
    """
    def __init__(self, chroma_db_path: Optional[str] = None) -> None:
        if chroma_db_path:
            if os.path.exists(chroma_db_path):
                client = chromadb.PersistentClient(
                    path = chroma_db_path,
                    settings = Settings(anonymized_telemetry=False)
                )
            else:
                client = chromadb.Client(settings = Settings(anonymized_telemetry=False))
        else:
            client = chromadb.Client(settings = Settings(anonymized_telemetry=False))
            
        self.chroma_collection = client.get_or_create_collection('docs')
        
        # Parent Chunks Store
        self.parent_doc_store = {}
    
    def _ingest_parent_docs(self, text: str, parent_num_tokens: int, parent_token_overlap: int) -> None:
        """
            Creates Parent chunks and stores them unique ids in a dictonary.
        """
        # --- 1. Obtain Parent Chunks ---
        parent_chunks = sentence_aware_splitter(text, parent_num_tokens, parent_token_overlap)
        
        # --- 2. Store Parent Chunks with unique ids in a dictonry
        for chunk in parent_chunks:
            # Get a unique id which remains same everytime for same text
            # This avoids chroma vectorstore getting polluted with same chunks with different ids
            parent_id = hashlib.sha256(chunk.encode()).hexdigest()
            self.parent_doc_store[parent_id] = chunk
    
    def _ingest_child_docs(
        self, 
        child_num_tokens: int, 
        child_token_overlap: int, 
        embed_model_name: str
        ) -> None:
        """
            Creates Child Chunks and stores them in Chromadb vectorstore.
        """
        if len(self.parent_doc_store) == 0:
            raise ValueError('Parent Chunks were not created.')
        
        # --- 1. Create Child Chunks
        for parent_id, parent_chunk in self.parent_doc_store.items():
            child_chunks = sentence_aware_splitter(parent_chunk, child_num_tokens, child_token_overlap)
            
            # --- 2. Store Child chunks embeddings in chromadb vector store
            for i, child_chunk in enumerate(child_chunks):
                child_id = hashlib.sha256(child_chunk.encode()).hexdigest()
                
                # generate embeddings
                embeddings = ollama.embed(embed_model_name, child_chunk)['embeddings']
                self.chroma_collection.add(
                    ids = child_id,
                    embeddings = embeddings,
                    metadatas={"parent_id": parent_id, "chunk_index": i},
                    documents=child_chunk
                )
    
    def _retrieve_docs(self, query: str, embed_model_name: str, n_results: int = 3) -> List[str]:
        """
            Retrieve relevant parent document chunks.
        """
        # --- 1. Embed user query ---
        query_embeddings = ollama.embed(embed_model_name, query)['embeddings']
        
        # --- 2. Semantic search between embeddings of user query and child chunks ---
        query_results = self.chroma_collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=['metadatas']
        )
        
        # --- 3. Retrieve parent chunks ids corresponding to retrieved child chunk ids ---
        retrieved_parent_ids = set()
        if query_results['metadatas']:
            for metadata_lists in query_results['metadatas']:
                for metadata in metadata_lists:
                    if 'parent_id' in metadata:
                        retrieved_parent_ids.add(metadata['parent_id'])
        
        # --- 4. Retrieve Parent Documents for downstream task
        retrieved_parent_docs = []
        for parent_id in retrieved_parent_ids:
            if parent_id in self.parent_doc_store:
                retrieved_parent_docs.append(self.parent_doc_store[parent_id])
        
        return retrieved_parent_docs

    def _debug(
        self, 
        text: str, 
        query: str, 
        embed_model_name: str, 
        num_tokens: tuple = (256, 128),
        token_overlap: tuple = (32, 16),
        n_results: int = 3
        ):

        print('='*50 + f'\n{'':<20}DEBUGGING\n' + '='*50)
        
        start = time.time()
        self._ingest_parent_docs(text, num_tokens[0], token_overlap[0])
        parent_ingest_time = time.time() - start
        
        print(
            f'\nParent number of Tokens:        {num_tokens[0]}'
            f'\nParent tokens overlap:          {token_overlap[0]}'
            f'\nParent Chunks Count:            {len(self.parent_doc_store)}'
            f'\nParent Ingestion Time:          {parent_ingest_time:.2f}s'
        )
        
        start = time.time()
        self._ingest_child_docs(num_tokens[1], token_overlap[1], embed_model_name)
        children_ingest_time = time.time() - start
        
        print(
            f'\nChild number of tokens:         {num_tokens[1]}'
            f'\nChild tokens overlap:           {token_overlap[1]}'
            f'\nChild Chunks Count:             {self.chroma_collection.count()}'
            f'\nChild Ingestion Time:           {children_ingest_time:.2f}s'
        )
        
        start = time.time()
        results = self._retrieve_docs(query, embed_model_name, n_results)
        context = ' '.join([f'- {result}\n\n' for result in results])
        query_retreival_time = time.time() - start
        print(
            f'\nLength of Context:              {len(context)}'
            f'\nQuery Retrieval Time:           {query_retreival_time:.2f}s'
            f'\n\nQuery: {query}'
            f'\nContext:\n{context}'
        )
