import pymupdf
import ollama
import re
from typing import Optional
import time
from datetime import datetime
import os
from text_splitters import sentence_aware_splitter

class Config:
    def __init__(self) -> None:
        self.chat_model = r'gemma3:4b'
        self.emb_model = r'snowflake-arctic-embed:33m'
        self.pdf_path = r'xyz.pdf'
        
        self.num_tokens = 1024
        self.token_overlap = 128
        self.num_sentences = 5
        
        self.options = {
            'temperature': 0.5,
            'top_k': 40,
            'top_p': 0.8,
            'num_ctx': 8192
        }
    
class PDFReaderPipe:
    def clean_text(self, text:str) -> str:
        # 1. Remove citations like "[1, 2, 3]"
        text = re.sub(r'\[\s*\d+(?:,\s*\d+)*\s*\]', '', text)
        
        # 2. Convert curly quotes to straight quotes
        text = re.sub(r'[“”]', '"', text)
        text = re.sub(r'[‘’]', "'", text)

        # 3. Handle hyphenated words at line breaks (e.g., "informa-tion")
        # This looks for a word-hyphen-newline pattern and removes the hyphen and newline.
        text = re.sub(r'([a-zA-Z])-\s*\n', r'\1', text)

        # 4. Replace all newlines with a single space
        text = re.sub(r'\n+', ' ', text)
        
        # 5. Normalize all whitespace to a single space, stripping leading/trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def read_pdf(self, filename:str)->Optional[str]:
        total_text = ''
        try:
            doc = pymupdf.open(filename)
            for page in doc:
                text = page.get_text()
                total_text += self.clean_text(text)
            doc.close()
        except Exception as e:
            print(f'Error in reading PDF: {e}')
            return None
        return total_text
            
class MapReduceSummarizer:
    def _map_function(self, chunk:str, config:Config) -> Optional[str]:
        """
            The 'MAP' step: summarizes a single chunk.
        """
        map_prompt = (
            'You are a helpful assistant. Please summarize the following text. '
            'Keep the summary concise and retain the most important details.\n\n'
            f'TEXT:\n---\n{chunk}\n---\nSUMMARY: '
        )
        try:
            response = ollama.generate(model=config.chat_model, prompt=map_prompt, options=config.options)['response']
            return response
        except Exception as e:
            print(f'Failed To Generate Chunk Summary.\n\n{e}')
    
    def _reduce_function(self, summaries: list[str], config:Config) -> Optional[str]:
        """
            The 'REDUCE' step: Consolidates the indivual chunk summaries into a single 
            cohhesive final summary.
        """  
        reduce_prompt = (
            'You are a helpful summarization assistant. You are given several summaries of a long document. '
            'Please combine these summaries into one final, coherent summary. '
            'Ensure all key points from the indivual summaries are included. '
            'Output the summary in proper Markdown format with bulltet key points.\n\n'
            f'SUMMARIES:\n---\n{'\n- '.join(summaries)}\n---\nFINAL SUMMARY: '
        )
        try:
            response = ollama.generate(model=config.chat_model, prompt=reduce_prompt, options=config.options)['response']
            return response
        except Exception as e:
            print(f'Failed To Generate Final Summary.\n\n{e}')
    
    def summarize(self, text: str, config: Config) -> Optional[str]:
        """
            Orchestrates the MapReduce Summarization process.
        """
        chunks = sentence_aware_splitter(text, config.num_tokens, config.token_overlap)
        # Map step: Summarize each chunk
        print('='*100 + f'\n{'':<35}CHUNKS SUMMARIZATION\n' +'='*100)
        start = time.time()
        chunk_summaries = []
        total_chunk_len = 0
        print(f'{'Chunk No.':<20}{'Chunk Length':<20}{'Summary Length':<20}{'Compression (%)':<20}{'Summarization Time (sec)'}')
        for i, chunk in enumerate(chunks):
            start_in = time.time()
            summary = self._map_function(chunk, config)
            if summary:
                chunk_summaries.append(summary)
            else:
                summary = ''
                print(f'WARNING: MAP step failed fo chunk {i+1}/{len(chunks)}...')
            end_in = time.time()
            
            chunk_len = len(chunk)
            total_chunk_len += chunk_len
            print(
                f'{i+1}/{len(chunks):<20}'
                f'{chunk_len:<20}'
                f'{len(summary):<20}'
                f'{round( (chunk_len - len(summary)) / chunk_len * 100, 2):<20}'
                f'{round(end_in-start_in, 2)}'
                )
        end = time.time()
        print(
            f'{'Total':<20}{len(text):<20}{total_chunk_len:<20}'
            f'{round((len(text) - total_chunk_len) / len(text) * 100, 2):<20}'
            f'{end-start:.2f}'
        )
        
        if not chunk_summaries:
            return 'Failed to generate only chunk summaries.'
        
        # Reduce step: Combine all chunk sumarries:
        start = time.time()
        final_summary = self._reduce_function(chunk_summaries, config)
        end = time.time()
        if final_summary:
            print(
            f'Final Summary Compression: {(len(text) - len(final_summary)) / len(text) * 100:.2f}\n'
            f'Final Summary Generation Time: {end-start:.2f}'
            )
            return final_summary
    
    def save_summary_as_markdown(self, summary: str, title: str, filename: Optional[str] = None):
        """
            Saves the summary to a Markdown File.
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f'Summary_{timestamp}.md'
        
        output = (
            f'# {title}\n\n' +
            f'## Final Summary\n' +
            summary
        )
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f'Summary successfully saved to: {filename}')
        except Exception as e:
            print(f'Failed to save markdown file: {e}')
            
if __name__ == '__main__':
    config = Config()
    pdfreader = PDFReaderPipe()
    summarizer = MapReduceSummarizer()
    
    text = pdfreader.read_pdf(config.pdf_path)
    summary = summarizer.summarize(text[:20000], config)
    summarizer.save_summary_as_markdown(
        summary,
        'Financial Summary',
        'temp/summary_temp.md',
    )
