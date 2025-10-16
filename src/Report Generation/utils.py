import os
import time
import requests
import logging
from langchain_community.graphs import Neo4jGraph
from langchain.agents import Tool
from llama_index.core.tools import FunctionTool
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Create logger
logger = logging.getLogger("lss_logger")
logger.setLevel(logging.INFO)
# Prevent duplicate handlers
if not logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    # File handler
    file_handler = logging.FileHandler("logs/lss_report.log", mode='a')
    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
oembed = OllamaEmbeddings(model="mxbai-embed-large")
vectordb = Chroma(
    persist_directory="mendeley_embed_db_2",
    embedding_function=oembed
)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

def process_dicom_file(file_path, model_name):
    url = 'http://223.195.54.164:8000//process_dicom/'
    
    with open(file_path, 'rb') as file:
        files = {'zip_file': file}
        data = {'model_name': model_name}
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()  # Raise an error for HTTP errors (4xx, 5xx)
            return response.json()  # Assuming the server returns JSON
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None

def generate_report(file_path, model_name):
    json_data = process_dicom_file(file_path, model_name)
    # print(json_data)
    if json_data is not None:
        return json_data


### GRAPH DATABASE
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
print(f"Connecting to Neo4j at {NEO4J_URI}...")

# Initialize connections
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)
def severity_based_relationship(
   node_type: str = "MedicalCondition",
    condition_name: str = "",
    severity_name: str = "",
    relationship_types=None,
    all_target_types=None
    ):
    """
    Build a Cypher query to get all related nodes for a MedicalCondition
    with a specific Severity, excluding the starting node type.

    :param node_type: Node type to start from (default "MedicalCondition")
    :param condition_name: The condition name to search for
    :param severity_name: The severity to filter on
    :param relationship_types: Relationships to follow
    :param all_target_types: Set of allowed node types (labels)
    :return: A dynamic Cypher query string
    """
    if relationship_types is None:
        relationship_types = ["CAUSES", "RECOMMENDS"]
    if all_target_types is None:
        all_target_types = {"Finding", "Procedure", "Severity", "DiscPathology", "AnatomicalStructure", "MedicalCondition"}

    # Build label filter clause excluding the current node_type
    filtered_labels = all_target_types - {node_type}
    where_clause = " OR ".join([f'"{label}" IN labels(target)' for label in filtered_labels])

    rel_clause = "|".join(relationship_types)

    cypher_query = f"""
    MATCH (n:{node_type})-[:HAS_SEVERITY]->(s:Severity)
    WHERE toLower(n.name) CONTAINS "{condition_name.lower()}"
      AND toLower(s.name) CONTAINS "{severity_name.lower()}"

    MATCH (n)-[r:{rel_clause}]->(target)
    WHERE {where_clause}

    RETURN DISTINCT n.name AS Condition,
                    s.name AS Severity,
                    type(r) AS Relationship,
                    target.name AS RelatedEntity,
                    labels(target) AS RelatedType
    """
    return cypher_query.strip()


# make for finding 2 without severity
def build_relationship_query(
    node_type: str = "Finding",
    name_contains: str = "",
    relationship_types=None,
    all_target_types=None
):
    """
    Build a dynamic Cypher query starting from a node of type `node_type`,
    filtering by partial name match, and returning connected nodes via specified relationships.

    :param node_type: Starting node label (e.g., 'Finding')
    :param name_contains: Partial name match string (e.g., 'dehydrated disc materials')
    :param relationship_types: Relationships to follow (e.g., ['ASSOCIATED_WITH', 'CAUSES', 'RECOMMENDS'])
    :param all_target_types: Set of valid target labels (e.g., ['MedicalCondition', 'Procedure', ...])
    :return: Dynamic Cypher query as a string
    """
    if relationship_types is None:
        relationship_types = ["CAUSES", "RECOMMENDS"]
    if all_target_types is None:
        all_target_types = {
            "Finding", "MedicalCondition", "Procedure",
            "Severity", "DiscPathology", "Anatomical Structure"
        }

    # Remove the starting node type from filter
    filtered_types = all_target_types - {node_type}
    where_clause = " OR ".join([f'"{label}" IN labels(target)' for label in filtered_types])
    rel_clause = "|".join(relationship_types)

    query = f"""
    MATCH (n:{node_type})
    WHERE toLower(n.name) CONTAINS "{name_contains.lower()}"

    MATCH (n)-[r:{rel_clause}]->(target)
    WHERE {where_clause}

    RETURN DISTINCT n.name AS StartNodeName, 
                    type(r) AS Relationship, 
                    target.name AS RelatedEntity, 
                    labels(target) AS RelatedType
    """
    return query.strip()

#findings 1
def KG_findings(node, conditionName, severity):
    # print(f"************ Medical Condition: {conditionName}, Severity: {severity} ************")
    FINAL_ANSWER = []
    query1 = severity_based_relationship(
        node_type= node,
        condition_name= conditionName,
        severity_name= severity
    )
    results1 = graph.query(query1)
    
    for result in results1:
        if result["RelatedEntity"] not in " ".join(FINAL_ANSWER):
            interm_resp = ""
            interm_resp = f"""'{result["Condition"]}' with severity: '{result["Severity"]}' {result["Relationship"]} '{result["RelatedEntity"]}'"""
            FINAL_ANSWER.append(interm_resp)

            query_2 = build_relationship_query(
            node_type=result["RelatedType"][0],
            name_contains=result["RelatedEntity"]
            )
            results_2 = graph.query(query_2)

            for result2 in results_2:

                interm_resp = f"""'{result2["StartNodeName"]}' {result2["Relationship"]} '{result2["RelatedEntity"]}'"""
               
                if result2["StartNodeName"] not in FINAL_ANSWER:
                    FINAL_ANSWER.append(interm_resp)
                break

    return FINAL_ANSWER[:4]


def graph_retrieval(query):
    concat_findings = []
    # query -> extreact spinal canal stenosis with severity -> kg -> RESULTS
    for level in query["stenosis_grading_info_findings"]:
        if 'Sever' in list(level.values()):
            # print("LEVEL:", level["level"])
            # call graph DB to get this levels stenosis severity
            resp_stenosis = KG_findings(node="MedicalCondition", conditionName="spinal canal stenosis", severity="secondary")
            # print(resp_stenosis)
            concat_findings.append(" ".join(resp_stenosis))
    
    # query -> disc with severity -< KG -> RESULTS
    for level in query["disc_bulge_info_findings"]:
        if "Normal" not in list(level.values()):
            # print("LEVEL:", level["level"])
            resp_disc = KG_findings(node="DiscPathology", conditionName="disc", severity="mild")
            # print(resp_disc)
            concat_findings.append(" ".join(resp_disc))
        
    # query -> spondylolisthesis with severity -< KG -> RESULTS
    for indx, levelinfo in enumerate(query["spondylolisthesis_info"]):
        if "Normal" not in levelinfo:
            # print("LEVEL:", indx+1)
            resp_spondylolisthesis = KG_findings(node="MedicalCondition", conditionName="Spondylolisthesis", severity="grade 1")
            # print(resp_spondylolisthesis)
            concat_findings.append(" ".join(resp_spondylolisthesis))
            
    # merge results -> return
    
    return "\n".join(list(set(concat_findings)))
    # extract 
##################
# Tool 1: Retriever
def retrieve_fn(query: str) -> str:
    
    retrieved_docs_base = retriever.invoke(query)
    results = "\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs_base)])
    return results

retriever_tool = FunctionTool.from_defaults(
    fn=retrieve_fn,
    name="Retriever",
    description="Retrieves documents based on a structured query"
)

# Tool 2: Evaluator
def evaluate_fn(query: str, context: str, llm) -> str:
    prompt = f"""Check if the following Retrieved Docs are relevant to the Content.
    
    Retrieved Docs: {query}
    Content: {context}
    
    Reply with only 'satisfied' or 'not satisfied'.
    """
    return llm.predict(prompt).strip().lower()

def evaluation_tool_fn(input: dict, llm) -> str:
    return evaluate_fn(input["retrieved_docs"], input["context"], llm)

evaluation_tool = FunctionTool.from_defaults(
    fn=evaluation_tool_fn,
    name="Evaluator",
    description="Checks if the retrieved docs satisfy the original query"
)
## RAG FUSION
def reciprocal_rank_fusion(results, k=5):
    
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)
    return [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]]

def rag_fusion(llm, query):
    # Multi Query: Different Perspectives
    query_template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_template)

    generate_queries = (
        prompt_rag_fusion  
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    try:
        retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
        retrieved_docs = retrieval_chain.invoke({"question": query})
        print("***************** Total Fusion Retrieved Docs *****************")
        print(len(retrieved_docs))
        
    except:
        retrieved_docs = None
    if retrieved_docs:
        results = "\n".join(
            line.strip()
            for doc in retrieved_docs
            for line in doc[0].page_content.strip().splitlines()
            if line.strip()
        )
        print("***************** Results from RAG Fusion *****************")
        print(results)
        return "\n".join([f"{i+1}. {doc[0].page_content}" for i, doc in enumerate(retrieved_docs)]), results

# Agent Loop
def agentic_rag_reasoning(llm, query: str, max_retries: int = 5):
    for step in range(max_retries):
        print(f"\n🔁 Reasoning Step #{step+1}")

        # Step 1: Structure the query
        # structured_query = query_structurer_tool(query)
        # print("Structured Query:", structured_query)
        
        # Step 2: Retrieve docs
        print("**********************Retrieving Documents from DB**********************")
        start_time = time.time()
        print("**********************Retrieval Query**********************:\n", " ".join(query))
        retrieved_docs = retriever_tool(" ".join(query))
        
        print("**********************Retrieved Docs**********************:\n", retrieved_docs)  # preview
        time_taken_retrieval = int(round(time.time() - start_time, 2))
        print("time_taken_retrieval:", time_taken_retrieval)
        # Step 3: Evaluate satisfaction
        print("**********************Reasoning Over Retrieved Documents**********************")
        
        evaluation_result = evaluation_tool({
            "retrieved_docs": retrieved_docs,
            "context": query
        }, llm)
        print("**********************Reasoning Results**********************:", evaluation_result)
        if 'not satisfied' not in evaluation_result.content:
            print("**********************✅ Agent is satisfied with the results.**********************")
            return retrieved_docs, time_taken_retrieval
    
    print("**********************❌ Agent failed to get satisfactory result.**********************")
    print("********************** Going with last Retrieved results **********************")
    return retrieved_docs, time_taken_retrieval
  
 
def summarizer_tool(text: str, llm):
    print("********************** Post Retrieval Summarization **********************")
    summary_prompt = f"""
        You are a medical domain expert with expertise in lumbar MRI medical report and clinical data interpretation.

        Given the following similar retrieved lumbar MRI medical reports:
        {text}

        Your task is to generate a clear and concise summary that captures the most clinically relevant information, relationships, and insights from the data.

        Strictly follow these rules:
        - Only output a single coherent paragraph.
        - Use formal and precise medical language.
        - Do not include headings, bullet points, or patient-identifiable information.
        - Avoid redundancy and focus on key biomedical connections and implications.

        Return only the summary paragraph without any explanation or formatting.
        """
    res = llm.predict(summary_prompt).strip()
    print("********************** Post Retrieval Summarization Done **********************")
    return res

 # Post Retrieval Processing
def post_retrieval_filtering_tool(retrieved_reports: str, llm):
    print("********************** Post Retrieval Filtering to Extract Synonyms and Writing Style **********************")
    report_prompt = f"""
    You are a medical language expert.
    
    Given the following retrieved radiology reports:
    {retrieved_reports}
    
    Your task is to extract and return the following information, without any explanation:
    
    - synonyms: medical terms and their alternative wordings found in the reports.
    - writing_style: writing style rules, formatting patterns, phrasing styles, and tone commonly used across the reports.
    
    Exclude all medical findings or patient-specific content. Focus strictly on reusable linguistic elements and writing structure."""
    res = llm.predict(report_prompt).strip()
    print("********************** Post Retrieval Filtering Done **********************")
    return res

import textwrap

def save_to_text_file(findings, analysis, filename="cv_analysis_output.txt", wrap_width=100):
    def wrap_preserve_newlines(text):
        # Wrap each line individually, preserving empty lines
        lines = text.strip().splitlines()
        wrapped = [textwrap.fill(line, width=wrap_width) if line.strip() else '' for line in lines]
        return '\n'.join(wrapped)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * wrap_width + "\n")
        f.write("📝 FINDINGS\n")
        f.write("=" * wrap_width + "\n")
        f.write(wrap_preserve_newlines(findings) + "\n\n")

        f.write("=" * wrap_width + "\n")
        f.write("🔍 FINAL ANALYSIS\n")
        f.write("=" * wrap_width + "\n")
        f.write(wrap_preserve_newlines(analysis) + "\n")

    print(f"✅ Output saved to {filename}")

# Formatting blocks
def disc_bulge_string(data):
    return ", ".join([
        f"({item['level']}: Herniation Present: {'Yes' if item['herniation_present'] else 'No'}, Condition: {item['herniation_condition']}, Ratio: {item['herniation_ratio']})"
        for item in data
    ])

def stenosis_string(data):
    return ", ".join([
        f"({item['level']}: SCS: {item['Spine Central Stenosis (SCS)']}, LFS: {item['Left Foraminal Stenosis (LFS)']}, RFS: {item['Right Foraminal Stenosis (RFS)']})"
        for item in data
    ])

def deformity_string(data):
    return ", ".join([
        f"({item['level']}: Wedge {item['Wedge deformity']} ({item['Wedge classification']}), "
        f"Biconcave {item['Biconcave deformity']} ({item['Biconcave classification']}), "
        f"{item['Spondylolisthesis']})"
        for item in data
    ])

def ap_distances_string(data):
    return ", ".join([
        f"({item['level']}: AP Distance = {item['Spinal Canal Anterior Posterior (AP) distance']} mm)"
        for item in data
    ])
    
# Version 2           
def build_query(json_data):
    """Builds the query string with optimized sentence generation."""
    query_parts = []
    levels = ["L5-S1", "L4-L5", "L3-L4"]
    spond_levels = ["L1", "L2", "L3", "L4", "L5"]
    disc_bulge_info = []
    foraminal_info = []
    angles = []
    spondylolisthesis_info = []
    stenosis_info= []
    deformities_info=[]
    lumbar_lordosis = []
    disc_bulge_info_findings = []
    stenosis_grading_info_findings = []
    deformity_findings = []
    ap_distances_findings = []
    
    
    if 'sagittal' in json_data and 'spondylolisthesis' in json_data['sagittal']:
        spondylolisthesis_measurements = json_data['sagittal']['spondylolisthesis'].get('measurements', {})
    else:
        spondylolisthesis_measurements = {}

    for level in spond_levels: 
        classification = spondylolisthesis_measurements[level].get("Classification")
        if "Normal" not in classification: # if classfication is not Normal
            spondylolisthesis_info.append(f"{level}: {classification}.")
        # else:
        #     spondylolisthesis_info.append(f"Spondylolisthesis classification for {level} is missing.")
    
    # DEFORMITIES MEASUREMENTS START ---------------------
    if 'sagittal' in json_data and 'vertebrae_deformities' in json_data['sagittal']:
        deformities_measurements = json_data['sagittal']['vertebrae_deformities'].get('measurements_info', {})
    else:
        deformities_measurements = {}
        
    for level in spond_levels:
        wedge_deformity = deformities_measurements[level]['Wedge Deformity']
        wedge_classification = deformities_measurements[level]['Wedge Classification']
        biconcave_deformity = deformities_measurements[level]['Biconcave Deformity']
        biconcave_classification = deformities_measurements[level]['Biconcave Classification']
        # spondylolisthesis
        classification = spondylolisthesis_measurements[level].get("Classification")
        if classification:
            spondylolisthesis_class = f"Spondylolisthesis at {level}: {classification}."
        else:
            spondylolisthesis_class = f"Spondylolisthesis classification for {level} is missing."
        #----------------
        deformity_findings.append({
            "level": level,
            "Wedge deformity": round(wedge_deformity, 2),
            "Wedge classification": wedge_classification,
            "Biconcave deformity": round(biconcave_deformity, 2),
            "Biconcave classification": biconcave_classification,
            "Spondylolisthesis": spondylolisthesis_class
        })
    print(deformity_findings)
    # DEFORMITIES MEASUREMENTS END ---------------------
    # deformities in query
    if "Normal" not in wedge_classification:
        deformities_info.append(f"{wedge_classification} Wedge deformity at level {level}")
    if "Normal" not in biconcave_classification:
        deformities_info.append(f"{biconcave_classification} Biconcave deformity at level {level}")
    #=======================
    
    for level in levels:
        if 'axial' in json_data and level in json_data['axial']:
            measurements_info = json_data['axial'][level]['axial_measurements'].get('measurements', {})

            #---------------- disc herination condition
            herniation_ratio = measurements_info["herniation_ratio"]

            if herniation_ratio > 1.1:
                condition = "Normal"
            elif 0.8 < herniation_ratio <= 1.1:
                condition = "Minor"
            else:  # herniation_ratio <= 0.8
                condition = "Severe"
            
            disc_herination_condition = condition
            disc_herniation = False if condition == "Normal" else True
            
            json_data['axial'][level]['axial_measurements']['measurements']['herniation_condition'] = disc_herination_condition
            
            disc_bulge_info_findings.append({
                "level": level,
                "herniation_present": disc_herniation,
                "herniation_condition": disc_herination_condition,
                "herniation_ratio": herniation_ratio})
            
            stenosis_grading_info_findings.append({
                "level": level,
                "Spine Central Stenosis (SCS)": measurements_info["stenosis_gradding"].get('CCS'),
                "Left Foraminal Stenosis (LFS)": measurements_info["stenosis_gradding"].get('LFS'),
                "Right Foraminal Stenosis (RFS)": measurements_info["stenosis_gradding"].get('RFS'),
            })
            # stenosis in  query
            if "Normal" not in measurements_info["stenosis_gradding"].get('CCS'):
                stenosis_info.append(f"""{measurements_info["stenosis_gradding"].get('CCS')} Spine Central Stenosis (SCS) at level {level}""")

            if "Normal" not in measurements_info["stenosis_gradding"].get('LFS'):
                stenosis_info.append(f"""{measurements_info["stenosis_gradding"].get('LFS')} Left Foraminal Stenosis (LFS) at level {level}""")
 
            if "Normal" not in measurements_info["stenosis_gradding"].get('RFS'):
                stenosis_info.append(f"""{measurements_info["stenosis_gradding"].get('RFS')} Right Foraminal Stenosis (RFS) at level {level}""")

            if "Normal" in measurements_info["stenosis_gradding"].get('CCS') and measurements_info["stenosis_gradding"].get('LFS') and measurements_info["stenosis_gradding"].get('RFS'):
                if f"No significant spinal canal narrowing noted" not in stenosis_info:
                    stenosis_info.append(f"No significant spinal canal narrowing noted")
            # ======================
                
            ap_distances_findings.append({
                "level": level,
                "Spinal Canal Anterior Posterior (AP) distance": measurements_info["SC_AP"]
            })
            #------------------------------------------
            
            left_formenial_dist = measurements_info.get('left_formenial_dist')
            right_formenial_dist = measurements_info.get('right_formenial_dist')
            
            if disc_herniation == True:
                if f"{disc_herination_condition} disc herniation/protrusion." not in disc_bulge_info:
                    disc_bulge_info.append(f"{disc_herination_condition} disc herniation/protrusion.")
            # elif disc_herniation is not None:
            #     disc_bulge_info.append(f"No significant spinal canal narrowing noted on {level}.")
            # else:
            #     disc_bulge_info.append(f"Data for {level} is missing or incomplete.")
            
            if left_formenial_dist is not None and right_formenial_dist is not None:
                foraminal_info.append(
                    f"Foraminal distances at {level}: Left Foraminal distance = {left_formenial_dist}mm, Right Foraminal distance = {right_formenial_dist}mm."
                )
            else:
                foraminal_info.append(f"Foraminal distance data for {level} is missing or incomplete.")
        # else:
            # disc_bulge_info.append(f"Data for {level} is missing or incomplete.")  # Handle missing levels
            # foraminal_info.append(f"Foraminal distance data for {level} is missing or incomplete.")
    # merge all finding for query
    if disc_bulge_info:
        disc_bulge_sentence = ", and ".join(disc_bulge_info)
        query_parts.append(f"{disc_bulge_sentence}")

    # if foraminal_info:
    #     foraminal_sentence = ", and ".join(foraminal_info)
    #     query_parts.append(f"Foraminal distances: {foraminal_sentence}")
        
    if spondylolisthesis_info:
        spondylolisthesis_sentence = ", and ".join(spondylolisthesis_info)
        query_parts.append(f"{spondylolisthesis_sentence}")
        
    if stenosis_info:
        stenosis_info_sentence = ", and ".join(stenosis_info)
        query_parts.append(f"{stenosis_info_sentence}")
        
    if deformities_info:
        deformities_info_sentence = ", and ".join(deformities_info)
        query_parts.append(f"{deformities_info_sentence}")

    lumbar_lordosis.append(json_data['sagittal']['lumbar_lordosis']['measurements'])
    angles.append(f"Sagittal vertebrae angle : LLA: {json_data['sagittal']['angles']['measurements']['LLA']} and LSA: {json_data['sagittal']['angles']['measurements']['LSA']}")
    query_parts.append(f"{json_data['sagittal']['lumbar_lordosis']['measurements']}")    
    # query_parts.append(f"LLA: {json_data['sagittal']['angles']['measurements']['LLA']} and LSA: {json_data['sagittal']['angles']['measurements']['LSA']}")
    query_parts.append(f"")
            
    return query_parts, foraminal_info, angles, spondylolisthesis_info, lumbar_lordosis, disc_bulge_info_findings, stenosis_grading_info_findings, deformity_findings, ap_distances_findings
