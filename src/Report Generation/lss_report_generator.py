import os
import json
import pandas as pd
import time
import re
import logging
import argparse
from typing import Dict, Any
from tqdm import tqdm
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama
from typing import Tuple
from Router import RoutingAgent
from agent_tools import *
from utils import *
from prompts import *

class LSSReportGenerator:
    def __init__(self, filename:str, folder_path: str, safe_path_name: str, llm_name: str,
                 prompt_template_findings: Any, prompt_template_analysis: Any,
                 model_name: str = "M", json_data: dict = None, retrieval_database: str = "RAG", fuse_retrieval: str = "KG_Agent", rag_type: str = "agentic"):
        """
        Initialize the LSS Report Generator with all necessary configurations.
        """
        self.filename = filename
        self.model_name = model_name
        self.json_data = json_data
        self.folder_path = folder_path
        self.safe_path_name = safe_path_name
        self.llm_name = llm_name
        self.prompt_template_findings = prompt_template_findings
        self.prompt_template_analysis = prompt_template_analysis
        self.retrieval_database = retrieval_database
        self.fuse_retrieval = fuse_retrieval
        self.rag_type = rag_type
        self.time_data = {}
        self.testing_predict_reports = []
        self.llm = ChatOllama(model=llm_name, base_url="http://localhost:11434")
        # Instantiate tools
        self.agentic_vector_tool = AgenticVectorDBTool(llm=self.llm, time_data=self.time_data)
        self.fusion_vector_tool = FusionVectorDBTool(llm=self.llm, time_data=self.time_data)
        self.knowledge_tool = KnowledgeDBTool(llm=self.llm, time_data=self.time_data)
        # Build routing agent
        self.agent = RoutingAgent(tools=[self.agentic_vector_tool, self.fusion_vector_tool, self.knowledge_tool])
        logger.info(f"Initialized LLM model: {llm_name}")

    def generate_findings(self, data: Dict[str, Any], context: str) -> str:
        logger.info("Generating Findings")
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template_findings)
        result = llm_chain.run(
            foraminal_info=data['foraminal_info'],
            angles=data['angles'],
            spondylolisthesis_info=data['spondylolisthesis_info'],
            lumbar_lordosis=data['lumbar_lordosis'],
            context=context,
            disc_bulge_info_findings=data['disc_bulge_info_findings'],
            stenosis_grading_info_findings=data['stenosis_grading_info_findings'],
            deformity_findings=data['deformity_findings'],
            ap_distances_findings=data['ap_distances_findings']
        )
        logger.info("Findings generated")
        if 'qwen' in self.llm_name:
            match = re.search(r'</think>\s*(.+)', result, re.DOTALL)
            # Extract and clean result
            if match:
                result = match.group(1).strip()
                    
        return f"{result}"

    def generate_analysis(self, data: Dict[str, Any], context: str) -> str:
        logger.info("Generating Final Analysis")
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template_analysis)
        result = llm_chain.run(
            foraminal_info=data['foraminal_info'],
            angles=data['angles'],
            spondylolisthesis_info=data['spondylolisthesis_info'],
            lumbar_lordosis=data['lumbar_lordosis'],
            context=context,
            disc_bulge_info_findings=data['disc_bulge_info_findings'],
            stenosis_grading_info_findings=data['stenosis_grading_info_findings'],
            deformity_findings=data['deformity_findings'],
            ap_distances_findings=data['ap_distances_findings']
        )
        logger.info("Analysis generated")
        return f"{result}"

    def generate_report(self) -> None:
        if self.json_data:
            item = self.json_data
        else:
            item = self._process_image(filename=self.filename, model_name=self.model_name)
        if item is not None:
            logger.info("Processing JSON file")
            filename = item['filename']
            patient_ID = int(os.path.basename(filename.replace(".zip", "")).lstrip("0") or "0")
            query = item['data']['query']
            self.time_data = {}
            context = self._retrieve_context(item['data'], query) 
            data = self._prepare_data(item['data'])
            start_time = time.time()
            findings = self.generate_findings(data, context)
            # findings = "LSS MRI"+"\n"+findings_
            print(findings)
            analysis = self.generate_analysis(data, context)
            print("Analysis:")
            print(analysis)
            self.time_data["llm_report_generation_time"] = int(round(time.time() - start_time, 2))

            self.time_data.update({
                "patient_ID": patient_ID,
                "retrieved_context": f"{context}",
                "predicted_notes": findings,
                "predicted_analysis": analysis,
                "foraminal_info": data['foraminal_info'],
                "angles": data['angles'],
                "spondylolisthesis_info": data['spondylolisthesis_info'],
                "lumbar_lordosis": data['lumbar_lordosis'],
                "disc_herniation_info_findings": data['disc_bulge_info_findings'],
                "stenosis_grading_info_findings": data['stenosis_grading_info_findings'],
                "deformity_findings": data['deformity_findings'],
                "ap_distances_findings": data['ap_distances_findings']
            })

            
            # Save to JSON file
            # out_json_path = f"{self.folder_path}/{self.safe_path_name}_patient_{patient_ID}_predictions.json"
            # with open(out_json_path, "w") as f:
            #     json.dump(self.time_data, f, indent=4)
            # logger.info(f"Saved predictions to {out_json_path}")
            # Save to excel file
            out_excel_path = f"{self.folder_path}/{self.safe_path_name}_predictions.xlsx"
            new_data_df = pd.DataFrame([self.time_data])
            if os.path.exists(out_excel_path):
                existing_df = pd.read_excel(out_excel_path)
                # Append new data
                updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            else:
                # No existing file, use new data
                updated_df = new_data_df

            # Save the updated DataFrame
            updated_df.to_excel(out_excel_path, index=False)     

    def _prepare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data['foraminal_info'] = ", ".join([f"({entry})" for entry in data['foraminal_info']])
        data['disc_bulge_info_findings'] = disc_bulge_string(data['disc_bulge_info_findings'])
        data['stenosis_grading_info_findings'] = stenosis_string(data['stenosis_grading_info_findings'])
        data['ap_distances_findings'] = ap_distances_string(data['ap_distances_findings'])
        return data

    def _process_image(self, filename: str, model_name: str):
        logger.info(f"Processing: {filename}")
        patient_ID = int(os.path.basename(filename).replace(".zip", "").lstrip("0") or "0")
        logger.info(f"patient_ID: {patient_ID}")
        file_path = os.path.join(filename)
        input_json = generate_report(file_path, model_name)
        # with open("other_data_jsons/rsna/diagnosis_patient_1097875334.json", 'r') as f:
        #     input_json = json.load(f)
        # patient_ID = "0"
        if not input_json: #if no json was generated, skip the patient
            logger.info(f"Error Processing Image [Segmentation]")
            return None
        
        json_data = input_json.copy()
        
        query, foraminal_info, angles, spondylolisthesis_info, lumbar_lordosis, disc_bulge_info_findings, stenosis_grading_info_findings, deformity_findings, ap_distances_findings = build_query(json_data)
    
        item = {
            "filename": filename,
            "patient_ID": patient_ID,
            "data": {
                 "query": query,
                "foraminal_info": foraminal_info,
                "angles": angles,
                "spondylolisthesis_info": spondylolisthesis_info,
                "lumbar_lordosis": lumbar_lordosis,
                "disc_bulge_info_findings": disc_bulge_info_findings,
                "stenosis_grading_info_findings": stenosis_grading_info_findings,
                "deformity_findings": deformity_findings,
                "ap_distances_findings": ap_distances_findings
                
            }
        }
        return item
    
    def _retrieve_context(self, data, query: str):
        if self.retrieval_database == "RAG":
            if self.rag_type == "agentic":
                logger.info("Using Agentic [Vector DB]")
                tool_inputs = {"agentic_vector_db": query}
                context = self.agent.run(tool_inputs=tool_inputs, tool_names=["agentic_vector_db"])
                
            elif self.rag_type == "fusion":
                logger.info("Using Fusion [Vector DB]")
                tool_inputs = {"fusion_vector_db": query}
                context = self.agent.run(tool_inputs=tool_inputs, tool_names=["fusion_vector_db"])
                
        elif self.retrieval_database == "KG":
            logger.info("Using Knowledge Graph retrieval")
            tool_inputs = {"knowledge_db": data}
            context = summarizer_tool(self.agent.run(tool_inputs=tool_inputs, tool_names=["knowledge_db"]), self.llm)
        
        elif self.retrieval_database == "Retrieval Fusion":
            if self.fuse_retrieval == "KG_Agent":
                logger.info("********************** [Agentic RAG [Vector DB + Knowledge Graph DB] **********************") 
                tool_inputs = {"agentic_vector_db": query, "knowledge_db": data}
                context = self.agent.run(tool_inputs=tool_inputs, tool_names=["knowledge_db", "agentic_vector_db"])
                logger.info(context)
                print("Context:")
                print(context)
                
            elif self.fuse_retrieval == "KG_RAG_Fusion":
                logger.info("********************** [KG + RAG Fusion] **********************")
                tool_inputs = {"fusion_vector_db": query, "knowledge_db": data}
                context = self.agent.run(tool_inputs=tool_inputs, tool_names=["knowledge_db", "fusion_vector_db"])
                logger.info(context)
        return context


def parse_args():
    parser = argparse.ArgumentParser(description="Run LSS Report Generator")
    parser.add_argument('--filename', type=str, help='Image file path', required=True)
    parser.add_argument('--llm', type=str, default='mistral', help='LLM model name')
    parser.add_argument('--retrieval', type=str, default='RAG', help='Retrieval type: RAG or KG')
    parser.add_argument('--fuse_retrieval', type=str, default='KG_Agent', help='If Fusion of Retrieval DBs, subtype: KG_Agent or KG_RAG_Fusion')
    parser.add_argument('--ragtype', type=str, default='agentic', help='RAG type: agentic or fusion')
    return parser.parse_args()


def main():
    args = parse_args()
    evaluation_folder = "evaluation"
    filename = args.filename
    llm_name = args.llm
    retrieval_database = args.retrieval
    fuse_retrieval = args.fuse_retrieval
    rag_type = args.ragtype

    safe_path = llm_name.replace(":", "_").replace("/", "_").replace("-", "_").replace(".", "_")
    output_path = os.path.join(evaluation_folder + f"_{retrieval_database}_{rag_type}", safe_path)
    os.makedirs(output_path, exist_ok=True)

    generator = LSSReportGenerator(
        filename=filename,
        folder_path=output_path,
        safe_path_name=safe_path,
        llm_name=llm_name,
        prompt_template_findings=prompt_template_findings,
        prompt_template_analysis=prompt_template_analysis,
        retrieval_database=retrieval_database,
        fuse_retrieval=fuse_retrieval,
        rag_type=rag_type
    )
    generator.generate_report()


if __name__ == "__main__":
    main()
