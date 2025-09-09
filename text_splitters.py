import ollama
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import List, Optional

def sentence_aware_splitter(text, num_tokens, token_overlap):
    if num_tokens <= token_overlap:
        raise ValueError("num_tokens must be greater than token_overlap")

    # Split the text into sentences
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    # Create a list of tuples, where each tuple contains the sentence and its word count
    sentence_data = [(s, len(word_tokenize(s))) for s in sentences]

    chunks = []
    current_sentence_idx = 0
    while current_sentence_idx < len(sentence_data):
        chunk_sentences = []
        chunk_token_count = 0
        end_sentence_idx = current_sentence_idx

        # --- Build a chunk by adding whole sentences ---
        # Greedily add sentences to the chunk until the token limit is reached.
        while end_sentence_idx < len(sentence_data):
            sentence, count = sentence_data[end_sentence_idx]
            
            # If a single sentence is longer than num_tokens, it becomes its own chunk.
            if not chunk_sentences and count > num_tokens:
                chunk_sentences.append(sentence)
                chunk_token_count += count
                end_sentence_idx += 1
                break

            # If adding the next sentence would exceed the token limit, stop.
            if chunk_token_count + count > num_tokens:
                break
            
            # Otherwise, add the sentence to the current chunk.
            chunk_sentences.append(sentence)
            chunk_token_count += count
            end_sentence_idx += 1
        
        # Add the completed chunk to the list of chunks.
        chunks.append(" ".join(chunk_sentences))

        # If we've processed all sentences, we're done.
        if end_sentence_idx >= len(sentence_data):
            break

        # --- Determine the start of the next chunk for overlap ---
        # To create an overlap, we step back from the end of the current chunk.
        overlap_token_count = 0
        next_start_idx = end_sentence_idx - 1
        
        # Keep moving the start index back until we have enough tokens for the overlap.
        while next_start_idx > current_sentence_idx and overlap_token_count < token_overlap:
            # We subtract 1 because next_start_idx is an index
            overlap_token_count += sentence_data[next_start_idx][1]
            next_start_idx -= 1
        
        # The next chunk starts at the calculated index.
        # We use max() to ensure the process always moves forward.
        current_sentence_idx = max(next_start_idx + 1, current_sentence_idx + 1)
            
    return chunks

def semantic_chunker(text: str, embedding_model: str, threshold_percentile: int = 95)->List[str]:
    """
        Splits text into semantically coherent chunks.
        - A low percentile (e.g., 5 or 10) will only split on the most drastic topic changes, creating fewer, larger chunks.
        - A high percentile (e.g., 50 or 75) is more sensitive and will split more often, creating more, smaller chunks.
    """
    # --- 1. Split text into indivual sentences ---
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return [text] # Not enough sentences to compare
    
    # --- 2. Generate embeddings for each sentence ---
    embeddings = [ollama.embeddings(embedding_model, sentence)['embedding']  for sentence in sentences]

    # --- 3. Calculate cosine similarity between consecutive sentences ---
    similarites = []
    for i in range(len(embeddings) - 1):
        emb1 = np.array(embeddings[i])
        emb2 = np.array(embeddings[i+1])
        similarites.append(
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        )
    
    # -- 4. Identify split points based on the threshold ---
    # The threshold is the value at the specified percentile of the similarity scores
    score_threshold = np.percentile(similarites, threshold_percentile)
    # any score below this threshold is a "split point"
    split_indices = [i+1 for i, sim in enumerate(similarites) if sim < score_threshold]
    
    
    # --- 5. Create Chunks --- 
    chunks = []
    start_idx = 0
    for end_idx in split_indices:
        chunk = " ".join(sentences[start_idx: end_idx])
        chunks.append(chunk)
        start_idx = end_idx
    
    # append last chunk
    chunks.append(sentences[start_idx:])
    return chunks
