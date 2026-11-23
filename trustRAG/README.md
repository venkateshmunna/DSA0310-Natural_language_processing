# trustRAG

**A Multimodal Hybrid RAG Framework for Domain-Specific Information Retrieval and Faithful Question Answering across Images and Videos**

trustRAG is a locally runnable academic capstone project for asking evidence-grounded questions about image and video collections. The project is being built in five stages.

## Complete application: Stages 1-5

- Upload multiple JPG, JPEG, PNG, WEBP, MP4, MOV, AVI and MKV files.
- Persist source media, extracted video frames and processing results in SQLite.
- Choose a video frame sampling interval from 1 to 60 seconds.
- Extract visible English text with EasyOCR and generate image descriptions with BLIP.
- Browse files, processing status, descriptions, OCR and timestamped video frames.
- Encode images/frames and text queries with the same CLIP model.
- Save normalized CLIP image vectors in a persistent FAISS inner-product index.
- Retrieve by BM25 keyword matching, CLIP semantic similarity, or normalized weighted hybrid ranking.
- In Chat, optionally search one or more selected files; leave the file selection empty to search all processed media.
- Inspect ranked image/frame evidence, filenames and video timestamps in Chat.
- Regenerate the visual index from Knowledge Base after adding or processing media.
- Send retrieved evidence to a locally available Ollama chat model.
- Require answers to cite retrieved sources and include video timestamps when applicable.
- Return an explicit insufficient-evidence response without calling Ollama when evidence is weak or absent.
- For UK traffic footage, scan the selected video's frames chronologically, upscale candidate text regions, and report confidence-filtered plate OCR with source-frame timestamps.
- Upload an evaluation CSV containing queries and relevant media asset IDs.
- Calculate measured Precision@K, Recall@K and MRR for BM25, Dense and Hybrid retrieval.
- Measure Faithfulness using reference-answer text overlap or an optional local Ollama classifier.
- Persist completed evaluation results and display summary tables, charts and per-query records.

## Requirements

- Python 3.10 through 3.13
- Windows, macOS or Linux
- Internet access on first processing run to download the EasyOCR and BLIP model files
- Ollama installed locally with at least one downloaded chat model
- Sufficient memory and disk space for PyTorch and local vision models
- Internet access for the optional first-time Hugging Face model downloads

## Install and run

From the project directory, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Before using answer generation, confirm Ollama is running and has a downloaded model:

```powershell
ollama serve
ollama pull llama3.2:3b
ollama list
```

Chat lists models currently returned by Ollama's local API. The default preference is `gemma2:2b`; if that model is unavailable, Chat selects the first returned model.

If PowerShell prevents virtual-environment activation, run Streamlit using the environment's executable:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Open the local URL printed by Streamlit in your browser.

PyTorch wheels are large. If pip cannot find a compatible Windows CPU-only wheel, install a CPU build using the command recommended by the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then run the requirements installation again.

## Local data and models

Original uploads, the SQLite database, extracted frames, FAISS index and its asset-ID map are stored under `data/` in the project folder. This directory is excluded from Git and remains on disk between application restarts. The first processed image/video downloads EasyOCR's English recognition model and the BLIP image-captioning model; building the visual index downloads `openai/clip-vit-base-patch32`. Models use the normal local Hugging Face cache. Processing and embedding run on the CPU.

The application currently processes each upload when you click **Save and process media**. Failed media remains saved with its error shown in the Knowledge Base. After indexing, select BM25, Dense (CLIP + FAISS) or Hybrid in Chat. BM25 can be used without a FAISS index; Dense and Hybrid require a current index. Chat retrieves supporting media and, when sufficient textual evidence exists, sends it to Ollama with instructions to avoid outside facts and cite sources. If evidence is weak or unavailable, trustRAG returns an explicit insufficient-evidence response without invoking Ollama. The Evaluation page reports only measured values from an uploaded labeled dataset.

For license-plate questions, select one or more files in Chat; leaving the file selection empty searches all processed media. Plate scanning runs only on selected videos. The reader is specialized for standard UK registrations, re-runs OCR on 3x–5x enlarged original-resolution candidate crops, and uses the crop-read confidence for recovered text. It caches confidence-filtered readings under `data/plate_cache/`. It returns OCR readings with frame previews and timestamps, not verified identities. Blurry, distant, occluded or non-UK plates can be missed or misread; review the displayed source frames.

## Evaluation dashboard

The Evaluation page does not ship benchmark numbers. It requires the user to provide ground-truth rows in a CSV:

| query | relevant_asset_ids | reference_answer |
|---|---|---|
| What revenue is shown? | `asset-id-1\|asset-id-2` | Revenue increased by 25 percent. |

`relevant_asset_ids` are the IDs shown for media assets in the SQLite database or the FAISS metadata. Separate multiple IDs with `|`, `;`, or commas.

The app calculates:

- Precision@K: relevant assets in the top K divided by K.
- Recall@K: relevant assets in the top K divided by all relevant assets for that query.
- MRR: mean reciprocal rank of the first relevant asset.
- Text-overlap Faithfulness: overlap between a reference answer and retrieved captions/OCR, divided by reference-answer tokens. It reports 0.0 when no reference answer is supplied.
- Ollama Faithfulness: an optional strict local-model classifier marks the retrieved answer as FAITHFUL or NOT_FAITHFUL.

Results are stored in `data/evaluation/`, remain available after restarting, and are never presented as scores until the current dataset has been retrieved and measured.

## Run tests

```powershell
python -m unittest discover -s tests -v
```
