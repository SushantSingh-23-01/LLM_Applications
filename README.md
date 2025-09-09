## LLM Applications

This is a repository aimed at implementing various applications of LLMs. The list is as follows:
1. **Retrieval Augmented Generation** (RAG) Pipeline.
2. **Document Summarization Pipeline**.
3. **Web Search Agent**.

All the applications invovle detailed debugging information displayed in terminal for easy tuning.


### Document Summarization

Implements Map Reduce Summarization Technique. 

**Steps**: 
- **Map Step**: Large Text is splitted into small text which fits within context window of LLM and then further summarized by it.
- **Reduce Step**: Collect summaries of chunks from map step and refine them into a single summary using LLM.

![Map Reduce Summarization.jpg](https://github.com/SushantSingh-23-01/LLM_Applications/blob/8d6065c85998eec24a58bc81cb9a1403873c4fa6/assets/Map_Reduce_summarization.jpg)

**Parameters Suggestion**:
- *Temperature*: 0.1 - 0.5
- *Top_K*: 10 - 50
- *Top_P*: 0.5 - 0.9
- *num_ctx*: Whatever GPU supports.
- *num_token*: As large as possible which allows chunks to fit within context window. 
- *num_ctx, num_tokens, tokens_overlap*: Tune them to minimize the number of chunks. Remember each chunk a LLM call is made.
