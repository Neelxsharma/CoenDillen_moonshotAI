# CoenAI V0.1

This repository contains the first local prototype of CoenAI, a source-grounded conversational portrayal of Dutch footballer Coen Dillen.

The prototype uses the smaller Qwen3 1.7B chat model and an embedding model through Ollama to reduce response time on modest hardware; the smaller chat model may be less capable than Qwen3 4B. It indexes reviewed notes in ChromaDB, retrieves passages relevant to a visitor's question, and asks the language model to answer using that evidence. It is retrieval-augmented generation (RAG), not a model trained to be Coen Dillen. The app instructs Qwen3 to use non-thinking mode and buffers the answer so marked thinking text can be removed before display. It replies briefly by default, expands when asked, and has no fixed word or token cap.

CoenAI is not Coen Dillen and does not have his memories or personal experiences.

## V0.1 contents

- `coenai_v0.1/chat_local.py` — terminal chat and evidence retrieval
- `coenai_v0.1/index_sources.py` — source chunking, embedding, and indexing
- `coenai_v0.1/sources/` — two short historical notes with source citations
- `coenai_v0.1/requirements.txt` — Python package requirements

The generated ChromaDB database and Python virtual environment are intentionally excluded; both are recreated locally.

## Run on Windows

Install [Ollama](https://ollama.com/download), then open PowerShell at the repository folder:

```powershell
ollama pull qwen3:1.7b
ollama pull qwen3-embedding:0.6b

cd .\coenai_v0.1
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python .\index_sources.py
python .\chat_local.py
```

Add or edit `.md` or `.txt` research notes inside `coenai_v0.1/sources/`, then rerun `index_sources.py`. Each note should clearly identify its source, URL, and date checked.

## Scope and limitations

- Text-only terminal prototype. No avatar, voice, hardware, or live internet lookup.
- Answers can only draw on the supplied notes and can still be mistaken; verify claims against their cited sources.
- The source collection is small and should be reviewed and expanded before public or museum use.
- Check source and model license terms before redistributing content or deploying the project.
