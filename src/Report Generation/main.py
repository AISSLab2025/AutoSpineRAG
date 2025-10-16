from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import uvicorn

from prompts import *
from lss_report_generator import LSSReportGenerator

app = FastAPI()

@app.post("/generate-report/")
async def generate_report(
    file: UploadFile,  # Select and upload the file
    llm_name: str = Form('mistral'),
    retrieval_database: str = Form('RAG'),
    fuse_retrieval: str = Form('KG_Agent'),
    rag_type: str = Form('agentic')
):
    try:
        base_directory = os.path.join("mendeley_dicom_testing", "dicom")
        filename = os.path.join(base_directory, file.filename)  # Only extract the file name (not saving it)
        
        safe_path = llm_name.replace(":", "_").replace("/", "_").replace("-", "_").replace(".", "_")
        output_path = os.path.join(f"evaluation_{retrieval_database}_{rag_type}", safe_path)
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

        return JSONResponse(content={"status": "success", "message": "Report generated", "output_folder": output_path})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# 🔥 Run with Uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
