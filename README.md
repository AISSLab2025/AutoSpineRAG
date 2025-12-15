>[![Paper](https://img.shields.io/badge/paper-OpenReview-6f42c1.svg)](https://openreview.net/forum?id=RKOKlm4e4U#discussion)📄 **[Read the Paper here](https://openreview.net/forum?id=RKOKlm4e4U#discussion)**

<!-- Top hero gallery: 1 large image, 1 large image, then a 2-column row -->

![Demo](figures\Multi Agentic Chatbot.gif)

<div align="center" style="max-width:980px; margin:0 auto 18px;">
  <figure style="margin:0 0 12px 0; width:100%">
    <img src="figures\figure architecture 1.png" alt="Hero - sagittal overview" style="width:100%; max-height:420px; object-fit:cover; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.12);" />
  </figure>

  <figure style="margin:0 0 12px 0; width:100%">
    <img src="figures\figure architecture 2.png" alt="Hero - axial overview" style="width:100%; max-height:420px; object-fit:cover; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.12);" />
  </figure>

  <div style="display:grid; grid-template-columns: repeat(2, minmax(360px, 1fr)); gap:12px; align-items:start;">
    <figure style="margin:0;">
      <img src="figures/figure report generation.png" alt="Segmentation overlay" style="width:100%; height:auto; object-fit:contain; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.10);" />
    </figure>
    <figure style="margin:0 auto; text-align:center; justify-self:center;">
      <img src="figures/figure results.png" alt="Measurement extraction" style="width:50%; max-width:360px; height:50%; object-fit:contain; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.10); display:block; margin:0 auto;" />
    </figure>
  </div>
</div>

# AutoSpineAI

AutoSpineAI integrates volumetric medical image segmentation with retrieval-augmented vision-language report generation to produce interpretable, clinically-aligned lumbar spine assessments and radiology-style reports.

**Key highlights**
- 3D-aware segmentation and measurement modules tuned for lumbar spine MRI/CT volumes
- Automated morphological and stenosis measurements for vertebral levels L1–S1
- Retrieval-Augmented Generation (RAG) pipeline combining image features with domain-specific clinical text for accurate, consistent reports
- Modular API and command-line entry points for inference and report generation

---

**Table of Contents**

1. **Overview**
2. **Repository Structure**
3. **Installation**
4. **Quick Start**
5. **Usage**
6. **Architecture & Design**
7. **Evaluation**
8. **Citations**

---

**Overview**

AutoSpineAI is a research-grade Computer-Aided Diagnosis (CAD) system for lumbar spine analysis. It processes 3D medical images (MRI/CT), performs level-wise segmentation and measurements, and generates structured radiology-style reports using a hybrid retrieval + generative language pipeline.

The project focuses on clinical interpretability: segmentation-derived measurements feed into a retrieval-augmented generation (RAG) model to ensure generated text is evidence-backed and consistent with measured values.

---

**Repository Structure**

- `src/Report Generation/` — modules and agents for converting measurements and image features into diagnostic reports (e.g., `lss_report_generator.py`, `agent_tools.py`, `main.py`).
- `src/Report Generation/prompts.py` — prompt templates and retrieval artifacts.
- `src/Report Generation/evaluator.py` — utilities to evaluate generated reports.
- `src/Segmentation/` — segmentation notebooks, models and server-side inference code.
	- `src/Segmentation/Server_side/app.py` — REST API / microservice for segmentation and measurement endpoints.
	- `src/Segmentation/Server_side/model_utils.py` — model helpers and inference wrappers.
	- `src/Segmentation/notebooks/` — exploratory notebooks for measurement and inference.

---

**Installation**

1. Clone the repository:

```bash
git clone https://github.com/your-org/AutoSpineAI.git
cd AutoSpineAI
```

2. (Recommended) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows
```

3. Install dependencies (if `requirements.txt` exists):

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, typical dependencies include:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-image opencv-python nibabel fastapi uvicorn transformers sentence-transformers
```

Note: GPU acceleration (CUDA) is recommended for segmentation model inference.

---

**Quick Start**

1. Run the segmentation server (example):

```bash
python src/Segmentation/Server_side/app.py
# or with uvicorn
uvicorn src.Segmentation.Server_side.app:app --reload --host 0.0.0.0 --port 8000
```

2. Generate a report from measurements (example):

```bash
python "src/Report Generation/main.py" --input measurements.json --output report.md
```

See `src/Report Generation/main.py` for available CLI options and examples.

---

**Usage & API**

- Segmentation endpoints (example):
	- `POST /segment` — Upload a volume (NIfTI, DICOM bundle, or .nii) and receive segmentation masks and per-level measurements.
	- `GET /status` — Health check.

- Report generation:
	- Provide measured features (JSON) and optionally image-derived embeddings; the RAG pipeline will retrieve relevant clinical snippets and produce a report in Markdown or JSON.

For integration, use the thin client wrappers in `src/Report Generation/agent_tools.py` and `Router.py`.

---

**Architecture & Design**

- Segmentation: volumetric networks (3D U-Net family variants) are used to segment intervertebral discs, vertebral bodies and canal regions. Post-processing extracts geometric measurements for each lumbar level.
- Retrieval: image and measurement embeddings are used to query a clinical knowledge store (index of domain text & exemplar reports).
- Generation: a transformer/decoder model is primed with retrieved context and measurement tokens to produce final text; the pipeline is tuned for factual consistency with measurements.

This modular separation (segmentation → measurement → retrieval → generation) makes it easy to swap models or adapters for research experiments.

---

**Evaluation & Reproducibility**

- Segmentation performance: DICE / IoU metrics across vertebral structures. Place evaluation scripts and corresponding test data in `src/Segmentation/`.
- Report quality: automatic metrics (BLEU / ROUGE), plus human evaluation for clinical accuracy.

To reproduce the baseline experiments, set up the environment, download the curated dataset (if applicable), and run the provided evaluation notebooks in `src/Segmentation/notebooks/`.

---

**📚 Citations**

If you use AutoSpineAI in your research, please cite the project and any underlying methods you rely on. Example BibTeX entries for foundational components are listed below.

```
@inproceedings{salem2025autospineai,
  title={AutoSpineAI: Lightweight Multimodal CAD Framework for Lumbar Spine MRI Assessments},
  author={Salem, Saied and Habib, Afnan and Raza, Mukhlis and Al-Huda, Zaid and Al-maqtari, Omar and Ertu{\u{g}}rul, Bilal and Y{\i}ld{\i}r{\i}m, {\"O}zal and Gu, Yeong Hyeon and Al-antari, Mugahed A},
  booktitle={IEEE-EMBS International Conference on Biomedical and Health Informatics 2025}
}
```
**📚 Related Work**

```
@article{al2025evaluating,
  title={Evaluating AI-powered predictive solutions for MRI in lumbar spinal stenosis: a systematic review},
  author={Al-Antari, Mugahed A and Salem, Saied and Raza, Mukhlis and Elbadawy, Ahmed S and B{\"u}t{\"u}n, Ertan and Aydin, Ahmet Arif and Aydo{\u{g}}an, Murat and Ertu{\u{g}}rul, Bilal and Talo, Muhammed and Gu, Yeong Hyeon},
  journal={Artificial Intelligence Review},
  volume={58},
  number={8},
  pages={221},
  year={2025},
  publisher={Springer}
}

@article{razamultimodal,
  title={Multimodal Knowledge-Infused VLM for Respiratory Disease Prediction and Clinical Report Generation},
  author={Raza, Mukhlis and Salem, Saied and Kwon, Hyunwook and Hussain, Jamil and Gu, Yeong Hyeon and Al-Antari, Mugahed A},
  journal={IEEE journal of biomedical and health informatics}
}

@inproceedings{raza2025medxpert,
  title={MedXpert-CAD: A Multimodal Multi-agentic System for Clinical Imaging Analysis via Model Context Protocol LLM-Driven Agentic Workflows},
  author={Raza, Mukhlis and Salem, Saied and Habib, Afnan and Abdulmahmod, Osamah and Kwon, Hyunwook and Hussain, Jamil and Al-antari, Mugahed A},
  booktitle={International Workshop on Agentic AI for Medicine},
  pages={55--64},
  year={2025},
  organization={Springer}
}

```


**Acknowledgements & Contact**

This project was created by the AISSLab team. For questions, datasets, or collaboration inquiries, please open an issue or contact the maintainers.
