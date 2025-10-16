from langchain.prompts import PromptTemplate

prompt_template_analysis = PromptTemplate(
      input_variables=["foraminal_info", "angles", "lumbar_lordosis", "context", "disc_bulge_info_findings", "stenosis_grading_info_findings, deformity_findings", "ap_distances_findings"],
      template="""You are tasked with generating a comprehensive medical report based on the provided with data. The report should follow the structure below:
  Technioque:
  The analysis is based on the patient's complete DICOM volume of lumbar MRI, utilizing T2-weighted images for sagittal views and T1-weighted images for axial views. The analysis consists of two sections: Vertebrae analysis and IVDs analysis.
  
  Vertebrae analysis:
  Provide a detailed single paragraph only, with heading "Vertebrae analysis", no bullets or headings for each of the Vertebrae levels: L5, L4, L3, L2, L1. Give space for each level paragraph starting.
  - Deformities: Include deformities values and classification for each vertebrae level.
  - Spondylolisthesis: Include spondylolisthesis infomation for each vertebrae level.
  - Sagittal Vertebrae Angles: Describe any angular abnormalities or changes in vertebral alignment using data from "{angles}".
  These are the each Vertebrae level information: "{deformity_findings}".
  
  IVDs analysis:
  Provide a detailed single paragraph only, with heading "IVDs analysis", for each of the IVDs levels: L5-S1, L4-L5, L3-L4.
  For each IVDs level, describe the following in a narrative form:
  - Disc Herniation: State disc herniation, condition, and ratio. "{disc_bulge_info_findings}".
  - Stenosis Grading: Include central canal stenosis (CCS), left foraminal stenosis (LFS), and right foraminal stenosis (RFS). "{stenosis_grading_info_findings}".
  - Foraminal Distances: State both left and right foraminal dist explicitly. "{foraminal_info}".
  - AP Distance: Include `Anterior Posterior (AP) distance`. "{ap_distances_findings}".
  - Lu0mbar Lordosis: Comment on the curvature based on the measurement "{lumbar_lordosis}".
  
  Impression:
  - Highlight the most significant abnormalities
  - Provide an overall assessment of the patient's condition.

  Recommendations:
  - If the condition is severe, recommend further evaluation or treatment by a specialist.
  - If the condition is mild or moderate, provide lifestyle recommendations.

  Rules:
  Do not generalize or summarize numerical values. All measurements, distances, angles, and ratios must be reported exactly as provided.

  Previous findings and terminology from similar cases to inspire language and writing structure: {context}.
  """
  )

# prompt_template_findings = PromptTemplate(
#       input_variables=["foraminal_info", "angles", "spondylolisthesis_info", "lumbar_lordosis", "context", "disc_bulge_info_findings", "stenosis_grading_info_findings", "deformity_findings", "ap_distances_findings"],
#       template="""You are tasked with generating a comprehensive lumbar MRI medical report based on the provided with data. The report should follow the structure below:

# Findings:
#   LSS MRI
#   Summarize the key findings in a concise one paragraph only, no bullets or headings for the provided data:
#   - Disc Herniation: Identify and mention levels with disc herniation and describe the extent based on the disc herniation information "{disc_bulge_info_findings}", but do not include the numerical ratio explicitly.
#   - Stenosis Grading: Describe the spinal canal and foraminal narrowing status for each level based on the stenosis grading "{stenosis_grading_info_findings}".
#   - Foraminal Distances: Describe foraminal narrowing or preservation using the underlying foraminal distance data "{foraminal_info}", without stating the exact measurements.
#   - Deformities Classification: Highlight classifications "{deformity_findings}", in descriptive terms only, avoiding numeric grades or values.
#   - Spondylolisthesis Classification: Highlight any abnormal classifications, in descriptive terms only, avoiding numeric grades or values.
#   - Lumbar Lordosis: Comment on the curvature based on the measurement "{lumbar_lordosis}", but do not mention the exact angle.
#   - Sagittal Vertebrae Angles: Describe any angular abnormalities or changes in vertebral alignment using data from "{angles}", again without stating the numerical angles.
  
# Important:
# - The following 'context' contains similar radiology reports that reflect the expected writing style, terminology, and typical length.
# - Carefully mimic the tone, phrasing, words, and structure observed in the context reports.
# - Keep the 'Findings' section short, precise, and clinically relevant.

# context:
# {context}.""")

prompt_template_findings = PromptTemplate(
    input_variables=["foraminal_info", "angles", "spondylolisthesis_info", "lumbar_lordosis", "context", "disc_bulge_info_findings", "stenosis_grading_info_findings", "deformity_findings", "ap_distances_findings"],
    template="""
You are a radiology language model assistant. Your task is to generate the 'Findings' section of a lumbar spine MRI report based on structured clinical observations provided below.

Your output should:
- Closely follow the tone, style, length, structure, and phrasing observed in the 'context' reports (see below).
- Be written in one short, clinically coherent paragraph.
- Avoid bullet points or headings.
- Avoid numerical values, angles, or ratios.
- Use concise, medically accurate terminology that matches the reference examples.

Carefully integrate the following content:
- **Disc Bulge or Herniation**: Based on "{disc_bulge_info_findings}", describe location and effect (e.g., thecal sac compression, nerve root encroachment), but avoid direct size measurements or levels unless necessary for clarity.
- **Spinal Canal and Foraminal Stenosis**: From "{stenosis_grading_info_findings}" and foraminal info "{foraminal_info}", describe whether narrowing is mild/moderate/severe and the likely structures affected.
- **Deformities & Alignment**: From "{deformity_findings}" and vertebral angle data "{angles}", describe any noticeable curvature changes, listhesis, or alignment issues.
- **Spondylolisthesis**: Summarize abnormalities from "{spondylolisthesis_info}" in natural language without numeric grading.
- **Lumbar Lordosis**: Describe lumbar curvature based on "{lumbar_lordosis}" in general terms (e.g., preserved, reduced, exaggerated).
- **Anterior-Posterior Distances**: If relevant, from "{ap_distances_findings}", describe implications without quoting measurements.

**Important Notes:**
- Use phrasing and word choice consistent with real-world clinical radiology reports.
- Integrate "Features of muscle spasm" only if supported by context or deformity/lordosis data.
- If no abnormalities are present, say so clearly using clinical tone (e.g., "No evidence of disc herniation or significant thecal sac compression noted").

Context (reference examples to guide phrasing and style):
{context}
""")