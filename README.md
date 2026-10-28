# CoenAI V0.1

This repository contains a local, source-grounded prototype portraying Dutch footballer Coen Dillen.

## How factual answers work

CoenAI uses Ollama to run the Qwen3 embedding model and ChromaDB to search the local notes. To avoid unsupported claims, the app only returns claim text explicitly marked under a `Verified fact` heading when the retrieved note is within a conservative relevance threshold. It appends the source title, filename, and URL from that note. If it cannot find a sufficiently relevant claim, it says so rather than asking a chat model to fill in the gap.

The chat-generation model is deliberately not used for factual answers in this version. The previous free-form generation produced conflicting, unsupported claims and invented attributions. This evidence-only behavior prefers a safe refusal over an answer when the note collection is incomplete. The current notes are very small, so the app will refuse many broad questions until more reviewed source notes are added.

CoenAI is an AI portrayal, not Coen Dillen himself, and does not have his memories or personal experiences.

## V0.1 contents

- `coenai_v0.1/chat_local.py` — terminal interface, relevance gate, verified-claim extraction, and source citations
- `coenai_v0.1/index_sources.py` — source chunking, embedding, and indexing
- `coenai_v0.1/sources/` — reviewed historical notes
- `coenai_v0.1/requirements.txt` — Python package requirements

The generated ChromaDB database and Python virtual environment are intentionally excluded; both are recreated locally.

## Run on Windows

Install [Ollama](https://ollama.com/download), then open PowerShell at the repository folder:

```powershell
ollama pull qwen3-embedding:0.6b

cd .\coenai_v0.1
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python .\index_sources.py
python .\chat_local.py
```

Add or edit research notes inside `coenai_v0.1/sources/`. Put answerable facts under a `## Verified fact` heading and include the source title, URL, and date checked. Then rerun `index_sources.py`.

## Scope and limitations

- Text-only terminal prototype. No avatar, voice, hardware, or live internet lookup.
- Answers reproduce reviewed claim text and citations; unsupported or weakly matched questions receive a refusal.
- The relevance threshold is conservative and calibrated to the current embedding model and collection. Changing the model or Chroma distance metric requires recalibration.
- The source collection is small and should be expanded and reviewed before public or museum use.
- Check source and model license terms before redistributing content or deploying the project.
