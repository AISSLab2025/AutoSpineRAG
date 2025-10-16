import time
import re
from utils import *

class AgenticVectorDBTool:
    name = "agentic_vector_db"
    description = "Use this tool to retrieve agentic results from the vector database"
    
    def __init__(self, llm, time_data: dict):
        self.llm = llm
        self.time_data = time_data

    def run(self, query: str) -> str:
        start_time = time.time()
        resp, retrieval_time = agentic_rag_reasoning(self.llm, query)
        reasoning_time = int(round(time.time() - start_time, 2))
        if resp:
            start_time = time.time()
            synonyms_writingStyle = post_retrieval_filtering_tool(resp.content, self.llm)
            # print(synonyms_writingStyle)
            before_text = re.search(r'^(.*?)\*\*writing_style:\*\*', synonyms_writingStyle, re.S)
            synonyms = before_text.group(1).strip() if before_text else ""
            after_text = re.search(r'\*\*writing_style:\*\*(.*)$', synonyms_writingStyle, re.S)
            writing_style = after_text.group(1).strip() if after_text else ""
            # summarized_context = summarizer_tool(resp.content, self.llm)
            # context = "Similar Findings:" + "\n" + resp.content + "\n" + "Summarized Findings:" + "\n" + summarized_context + "\n" + synonyms_writingStyle
            # context = "Similar Findings:" + "\n" + resp.content + "\n" + "Expected synonyms and writing style:" + "\n" + synonyms_writingStyle
            # context = writing_style
            # context = synonyms_writingStyle
            context = resp.content + "\n" + synonyms_writingStyle
            # context = resp.content
            post_time = int(round(time.time() - start_time, 2))
            self.time_data.update({
                "agentRag_retrieval_query": f"{query}",
                "agentRag_retrieved_docs": f"{resp}",
                "agentRag_retrieval_time": retrieval_time,
                "agentRag_reasoning_time": reasoning_time,
                "agentRag_post_retrieval_time": post_time
            })
        else:
            context = ""
        return context

class FusionVectorDBTool:
    name = "fusion_vector_db"
    description = "Use this tool to retrieve fusion results from the vector database"
    
    def __init__(self, llm, time_data: dict):
        self.llm = llm
        self.time_data = time_data
        
    def run(self, query: str) -> str:
        start_time = time.time()
        retrieved_docs, context = rag_fusion(self.llm, query)
        context = summarizer_tool(context, self.llm)
        fusionRag_retrieval_time = int(round(time.time() - start_time, 2))
        self.time_data.update({
                "fusionRag_retrieval_query": f"{query}",
                "fusionRag_retrieved_docs": f"{retrieved_docs}",
                "fusionRag_retrieval_time": fusionRag_retrieval_time
            })
        return context
    
class KnowledgeDBTool:
    name = "knowledge_db"
    description = "Use this tool to retrieve from the knowledge base"

    def __init__(self, llm, time_data: dict):
        self.llm = llm
        self.time_data = time_data
        
    def run(self, data: dict) -> str:
        start_time = time.time()
        context = graph_retrieval(data)
        kgRag_retrieval_time = int(round(time.time() - start_time, 10))
        if context:
            start_time = time.time()
            # context = summarizer_tool(context, self.llm)
            kgRag_summarization_time = int(round(time.time() - start_time, 2))
            self.time_data.update({
                    "kgRag_retrieved_data": f"{context}",
                    "kgRag_retrieval_time": kgRag_retrieval_time,
                    "kgRag_summarization_time": kgRag_summarization_time
                })
        return context