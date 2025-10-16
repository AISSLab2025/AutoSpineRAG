# 🧠 LSS Report Generator

AI-based tool for generating diagnostic findings and analyses from spine-related imaging data using local large language models (LLMs) like Mistral or LLaMA via the Ollama framework. This project supports advanced reasoning strategies like Retrieval-Augmented Generation (RAG), including both agentic and fusion types.

---

## 🚀 Features

- ✅ **LLM-based Findings & Analysis Generation**
- 📚 **Agentic and Fusion RAG Support**
- 🔍 **Knowledge Graph or RAG Contextual Retrieval**
- 📤 **Excel Report Export for Each Patient**
- ⚙️ **Modular and Testable Codebase**
- 🧾 **Command-Line Interface (CLI) Support**
- 🪵 **Logging to Console and File**

---

## 📁 Directory Structure

```
project_root/
├── data/
│   └── JSON_data_for_testing_dicom_V3_deformities.json
├── logs/
│   └── report_generation.log
├── results/
│   └── [auto-generated results per LLM/RAG type]
├── src/
│   ├── main.py
│   ├── utils.py
│   └── prompts.py
├── run.sh
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone
cd lss-report-generator
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS
# venv\Scripts\activate   # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 How to Run

### Using Python CLI

```bash
python3 src/main.py --llm mistral --retrieval RAG --ragtype agentic
```

### Using Bash Script

```bash
chmod +x run.sh
./run.sh
```

---

## 🎛️ CLI Parameters

| Argument      | Description                           | Default   |
|---------------|---------------------------------------|-----------|
| --llm         | Name of LLM model (e.g., mistral)     | mistral   |
| --retrieval   | Retrieval strategy: RAG or KG         | RAG       |
| --ragtype     | RAG subtype: agentic or fusion        | agentic   |

---

## 📤 Output

- **Log File:** `logs/report_generation.log`
- **Excel Report:**  
  `results_RAG_agentic/mistral_patient_predictions.xlsx`  
  Each row contains:
  - Patient ID
  - Predicted findings
  - Final analysis
  - Time metrics for each step

---

## 🧠 Supported Models

You can run any local LLM supported by Ollama:

- Gemma3
- mistral
- llama3
- phi3
- Qwen3

## Supported Databases

You can run any Database of the following:

- Vector DB
- Knowledge Graph DB

## 🛠️ Code Overview
**src/main.py**:
Main entry point. Handles CLI arguments and API requests, orchestrates report generation using the LSSReportGenerator class.

**src/utils.py**:
Utility functions for DICOM processing, retrieval (RAG/KG), logging, and database connections.

**src/prompts.py**:
Prompt templates for findings and analysis generation.

**Data & Results**:
Place DICOM files in the appropriate data folder.
Generated reports and logs are saved in **results/** and **logs/**.