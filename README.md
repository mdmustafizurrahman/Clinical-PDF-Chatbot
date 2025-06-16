#  Clinical PDF Chatbot (FLAN-T5 + ClinVec)

##  Assumptions

1.  The user will upload **PDF** files only. (DOC, DOCX, TXT could be added with additional parsers.)
2.  Currently, it supports one pdf at a time to be uploaded, processed and answered from.
3.  The user will ask questions that are expected to be **answerable from the uploaded document**.

---

##  Model Information

-  **LLM**: `google/flan-t5-base`  instruction-tuned for QA tasks, small and CPU-friendly
-  **Clinical embeddings**: `ClinVec_phecode.csv` + `ClinGraph_nodes.csv` (Harvard MIMS)

## Why Clinical embedding?

- **Domain-Specific Knowledge**
ClinVec captures semantic relationships between clinical concepts (e.g., diagnoses, labs, medications) from real-world EHR vocabularies.

- **Enhanced Retrieval Relevance**
Using embeddings of clinical codes (like PheCodes or ICD10) helps retrieve contextually similar terms beyond exact matches — boosting recall.

- **Contextualize User Queries**
If a user refers to a condition or code, ClinVec can find related concepts even if not mentioned explicitly in the document.

- **Bridge Vocabulary Gaps**
ClinVec links vocabularies like SNOMED, RxNorm, LOINC — enabling interoperability across different clinical coding systems.

- **Lightweight and Efficient**
Embeddings are precomputed and compact (128D), making them fast to index with FAISS and easy to integrate alongside text-based RAG.

##  Clinical PDF Chatbot (FLAN-T5 + ClinVec)

This project is an interactive biomedical chatbot that:
- Accepts biomedical PDFs
- Processes and chunks the content
- Embeds it using sentence-transformers
- Retrieves relevant clinical codes using ClinVec embeddings
- Generates answers using an open LLM (`FLAN-T5-1.1B-Chat`)
- Tracks latency, and answer length

---

##  Features

-  Upload and chat over biomedical PDFs
-  Retrieve relevant chunks using text and clinical code semantics
-  Powered by FLAN-T5-1.1B-Chat for chat-style answers
-  Uses `ClinVec_phecode.csv` and `ClinGraph_nodes.csv` for clinical concept similarity
-  Built-in evaluation: latency, and token count
-  Export chat metrics to CSV

---

##  Installation

1. Clone or download the project
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Place the following files in `data/` directory:
   - `ClinVec_phecode.csv`
   - `ClinGraph_nodes.csv`

---

##  How to Run

```bash
streamlit run main.py
```

---

##  Directory Structure

```
app/
 embedding.py          # Embeds PDFs + ClinVec codes
 pdf_reader.py         # Extracts and chunks PDF text
 qa_engine.py          # FLAN-T5-based answer generation

main.py                   # Streamlit app interface
data/
 ClinVec_phecode.csv
 ClinGraph_nodes.csv
```

---

##  How to Interact

- Upload a PDF file (e.g., article, clinical report)
- Ask biomedical questions (e.g., "What is the risk of PheCode:008?")
- The app retrieves text + code context, generates answer

---

## Evaluation Approach

This chatbot application incorporates both human feedback and automated evaluation metrics to assess answer quality and guide continuous improvement.

### Business Metrics Logged
- **Latency**: Time taken to generate a response.
- **Answer Length**: Number of words in the response.
- **User Feedback**: Collected via thumbs up/down (Yes/No radio).

### Automated Evaluation Metrics

#### 1. Faithfulness (via Natural Language Inference)
Checks if the answer is grounded in the retrieved context using a pretrained entailment classifier (`roberta-large-mnli`).

- **Labels**:
  - `ENTAILMENT`: The answer is directly supported by the context ✅
  - `NEUTRAL`: The answer is plausible but not clearly supported ⚠️
  - `CONTRADICTION`: The answer contradicts the context ❌
- **Score**: Confidence score (range 0.0 – 1.0)

#### 2. Relevance (via Sentence Embedding Similarity)
Measures how semantically close the answer is to the question using cosine similarity between embeddings.

- **Scale**:
  - **> 0.8** → Highly relevant ✅
  - **0.5–0.8** → Moderately relevant ⚠️
  - **< 0.5** → Low relevance ❌

### Use for Improvement

- Flag answers with low faithfulness or relevance for review.
- Analyze metrics over time to tune prompts, improve retrieval, or switch models.
- Correlate low scores with negative user feedback to prioritize refinements.
                 |

All metrics are logged in memory and optionally exported to `chatbot_metrics.csv`.

---

##  Exporting Results

Click the **"Export metrics as CSV"** button in the UI after chatting to save interaction logs.

---

##  Acknowledgments

- [ClinVec embeddings](https://github.com/mims-harvard/Clinical-knowledge-embeddings) from Harvard MIMS
- [FLAN-T5](https://huggingface.co/google/flan-t5-base) open instruction-tuned LLM

---

##  Optional Ideas

- Swap FLAN-T5 for another LLM (e.g., Zephyr, Mistral)
- Add long-term chat memory
- Add multi-code similarity retrieval

---

##  System Requirements (Tested on macOS)

This project has been tested and works on the following macOS configuration:

```text
Darwin US_DR7V4GVH7R 24.5.0 Darwin Kernel Version 24.5.0: Tue Apr 22 19:54:26 PDT 2025; root:xnu-11417.121.6~2/RELEASE_ARM64_T8112 arm64
ProductName:    macOS
ProductVersion: 15.5
BuildVersion:   24F74
```

-  Apple Silicon (M1/M2/M3) compatible
-  Python 3.10 (via `venv`)
-  Supports MPS backend for PyTorch (CPU fallback recommended for large models)
-  If using MPS, watch for memory ceiling (~18GB limit on macOS)

> You may want to use:  
> `mp.set_start_method("spawn", force=True)` in `main.py` to avoid semaphore leaks on macOS.


---

##  Hardware Overview (Development Machine)

Tested on:

```text
Model Name: MacBook Air
Model Identifier: Mac14,2
Model Number: Z15S000D2LL/A
Chip: Apple M2
Total Number of Cores: 8 (4 performance and 4 efficiency)
Memory: 16 GB
System Firmware Version: 11881.121.1
OS Loader Version: 11881.121.1
```

This configuration has been used to run and test the chatbot locally, including:
- FAISS indexing
- Streamlit interface
- FLAN-T5 model (CPU fallback)
- PDF chunking and context retrieval

>  Tip: Use `sentence-transformers` and `FLAN-T5` with `torch_dtype=torch.float32` for smooth CPU execution on M2 chips.

---

## How to Download ClinVec Data

To enable clinical concept retrieval, this project uses embeddings and metadata from the [Clinical Knowledge Embeddings (ClinVec)](https://github.com/mims-harvard/Clinical-knowledge-embeddings) project by Harvard MIMS.

### Required Files

Download the following files and place them into a `data/` folder in your project root:

- `ClinVec_phecode.csv`
- `ClinGraph_nodes.csv`

You can download these from Harvard Dataverse:

- Dataset DOI: https://doi.org/10.7910/DVN/Z6H1A8
- Direct page: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Z6H1A8

These files provide:
- Precomputed clinical concept embeddings (128 dimensions)
- Mappings to clinical codes and terms from PheCode, ICD10, SNOMED, RxNorm, etc.
