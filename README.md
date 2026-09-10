# RAG Baseline App

A local "chat with documents" app. It reads files from `data/`, splits them into chunks, embeds those chunks, stores them in a vector database, and answers questions grounded in those documents using a local LLM.

## How to run

```bash
# 0. (optional) make Poetry create .venv/ inside this folder, one-time setting
poetry config virtualenvs.in-project true

# 1. Install deps
poetry install

# 2. Make sure Ollama is running and models are pulled
ollama serve
ollama pull nomic-embed-text
ollama pull qwen3:4b

# 3. Build the index from data/
poetry run python -m app.vector_store

# 4. (optional) sanity-check retrieval on its own
poetry run python demo_vector_check.py "What is RAG?"

# 5. Chat (terminal loop)
poetry run python -m app.main

# 6. Or run it as a REST API instead
poetry run uvicorn app.main:app --reload
```

Type a question, get an answer. Type `exit` to quit the terminal chat loop.

### REST API

```bash
curl -X POST http://127.0.0.1:8000/ingest

curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## Chunking strategy

Fixed-size character chunking (`CHUNK_SIZE=800` chars, `CHUNK_OVERLAP=120` chars), implemented in `app/ingestion.py`. Chosen because it's simple, has no dependencies, and works on any document regardless of structure — no need to detect sentence or paragraph boundaries. The 120-char overlap keeps an idea from being fully lost when it straddles a chunk boundary.

Trade-off: it can still cut a sentence in half mid-thought, which a paragraph-aware or semantic chunking approach would avoid, at the cost of being more complex to implement (see Bonus #1 below for a direct comparison).

## Embedding model & vector database

- **Embedding model**: `nomic-embed-text`, run locally via Ollama.
- **Vector database**: ChromaDB, local and persistent (`chroma_db/` folder), no server to run.
- **Generation model**: `qwen3:4b`, run locally via Ollama.

---

## Bonus challenges

### 1. Compare two chunking strategies

`compare_chunking.py` runs the existing fixed-size chunker and a new paragraph-aware chunker (groups whole paragraphs up to `CHUNK_SIZE`, only falling back to fixed-size splitting if a single paragraph is longer than that) on every file in `data/`:

```bash
poetry run python compare_chunking.py
```

| File | Fixed-size chunks | Paragraph-aware chunks |
|---|---|---|
| 001_Setting_Up_a_Mobile_Device_for_Company_Email.txt | 4 | 4 |
| 002_Resetting_a_Forgotten_PIN.txt | 3 | 3 |
| 003_Configuring_VPN_Access_for_Remote_Workers.txt | 4 | 4 |
| 004_Troubleshooting_Issues_with_Microsoft_Office.txt | 5 | 4 |
| 005_Setting_Up_a_Conference_Call_on_Cisco_Webex.txt | 4 | 4 |
| 006_Creating_a_Backup_of_Important_Files.txt | 4 | 4 |
| 007_Troubleshooting_Issues_with_Company-Issued_Tablets.txt | 4 | 4 |
| 008_Setting_Up_a_Secure_Wireless_Network.txt | 4 | 4 |
| 009_Resetting_a_Jammed_Printer.txt | 4 | 4 |
| 010_Configuring_Email_on_an_Android_Device.txt | 4 | 5 |

The paragraph-aware strategy produces the same chunk count as fixed-size on most files, one fewer on `004` (it merges more content per paragraph before splitting), and one more on `010` (it ends a chunk earlier rather than merge across a paragraph boundary). Overall it trades a slightly different chunk count for never splitting a paragraph mid-sentence.

### 2. Compare two vector databases

`vector_store_qdrant.py` mirrors `vector_store.py` but stores vectors in Qdrant instead of ChromaDB. `compare_vector_dbs.py` builds/queries both and reports timing.

```bash
poetry add qdrant-client
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
poetry run python -m app.vector_store
poetry run python -m app.vector_store_qdrant
poetry run python compare_vector_dbs.py "What is RAG?"
```

Actual result (40 chunks indexed in both):

```
=== ChromaDB (238.0 ms) ===
  source=006_Creating_a_Backup_of_Important_Files.txt distance=0.5023
  source=006_Creating_a_Backup_of_Important_Files.txt distance=0.5025
  source=006_Creating_a_Backup_of_Important_Files.txt distance=0.5267
  source=006_Creating_a_Backup_of_Important_Files.txt distance=0.6129

=== Qdrant (29.2 ms) ===
  source=006_Creating_a_Backup_of_Important_Files.txt score=0.7489
  source=006_Creating_a_Backup_of_Important_Files.txt score=0.7488
  source=006_Creating_a_Backup_of_Important_Files.txt score=0.7366
  source=006_Creating_a_Backup_of_Important_Files.txt score=0.6935
```

Same 4 chunks, same order, from both — retrieval is correct either way. Differences: Qdrant was ~11x faster on this small dataset; ChromaDB reports distance (lower = closer) while Qdrant here reports cosine score (higher = closer), so the numbers aren't on the same scale; and ChromaDB persists automatically to a local folder, while Qdrant's data lives only inside its Docker container unless you mount a volume (`-v "$(pwd)/qdrant_storage:/qdrant/storage"`).

### 3. Print retrieved chunks before the answer

Done in `app/main.py`'s chat loop — each retrieved chunk's source, distance, and a text preview print before `Answer:`.

### 4. "Not found" fallback when no strong match

Done in `app/pipeline.py` via `DISTANCE_THRESHOLD` — confirmed working below (see "what is rag?" on the IT-support dataset, where no chunk falls under the threshold).

---

## Reflection

Building the pipeline stage by stage made the whole idea of RAG click in a way just reading about it didn't. Wiring `ingestion.py` → `embeddings.py` → `vector_store.py` for the offline half, then `retriever.py` → `generator.py` for the online half, and finally joining both in `pipeline.py`, made it obvious where a bad answer actually comes from: a chunking problem, a retrieval miss, or a generation problem, instead of one big mystery box. Keeping every tunable value in `config.py` also paid off — adjusting `CHUNK_SIZE`, `TOP_K`, or the distance threshold to see how answers changed was a one-line edit instead of a hunt through files. Using `demo_vector_check.py` to check retrieval on its own, before ever calling the LLM, was the single most useful step: it caught weak or irrelevant retrieval early, before it could be masked by the generator confidently answering anyway.

The harder part than expected was the "not covered" case, and a related surprise: `qwen3:4b`'s default "thinking" mode occasionally returned a completely empty answer on a harder question, even though the correct chunks were retrieved. That took some tracing to realize it wasn't a retrieval or prompt problem at all, but the model spending its whole token budget on hidden reasoning before ever writing a final answer. Deciding when a question is genuinely outside the documents versus just phrased differently from the source text also isn't obvious from a single top-1 distance score, and a fixed threshold doesn't generalize well across question types.

For an improvement, I'd add re-ranking: retrieve a wider top-k with the embedding model, then use a cross-encoder (or the LLM itself) to re-score and reorder those candidates before generation. That should catch cases where the right chunk is retrieved but not ranked first, without touching the chunking or embedding stages at all.

---

## Test Log

Ran via `poetry run python -m app.main` on the IT-support ticket dataset (001–010).

### Q1 (on-topic)
**Question:** how to configuring email?
**Retrieved:** 010_Configuring_Email_on_an_Android_Device.txt (0.5015), 010_Configuring_Email_on_an_Android_Device.txt (0.5019), 001_Setting_Up_a_Mobile_Device_for_Company_Email.txt (0.5627), 010_Configuring_Email_on_an_Android_Device.txt (0.5933)
**Answer:** To configure email on an Android device: open the Email app, tap "Add account," select "Email" or "Corporate" as the account type, enter your email address and password, then configure server settings (the device may auto-configure; if needed, enter manual settings), and configure outgoing server settings.
**Sources:** 001_Setting_Up_a_Mobile_Device_for_Company_Email.txt, 010_Configuring_Email_on_an_Android_Device.txt

### Q2 (on-topic)
**Question:** how to confguring vpn access for remote workers?
**Retrieved:** 003_Configuring_VPN_Access_for_Remote_Workers.txt (0.2524, 0.5408, 0.6099, 0.6755) — all four top chunks from the same file
**Answer:** To configure VPN access for remote workers: download and install the VPN client from the company's software portal, configure a new connection by selecting "Company VPN" as the connection type, enter the company's VPN server address and login credentials, save the connection settings, click "Connect" (may require two-factor authentication), and verify the connection by accessing company resources and checking for a company-assigned IP address.
**Sources:** 003_Configuring_VPN_Access_for_Remote_Workers.txt

### Q3 (on-topic)
**Question:** how to creating a backup file?
**Retrieved:** 006_Creating_a_Backup_of_Important_Files.txt (0.4651, 0.5155, 0.5460, 0.6475) — all four top chunks from the same file
**Answer:** Identify important files, choose a backup method (external hard drive, cloud service like OneDrive/Google Drive, network share, or USB drive), connect your backup device, copy files to the backup device, verify the backup, and schedule regular backups.
**Sources:** 006_Creating_a_Backup_of_Important_Files.txt

### Q4 (on-topic)
**Question:** how to set up plan for wireless network?
**Retrieved:** 008_Setting_Up_a_Secure_Wireless_Network.txt (0.4568, 0.6994, 0.7132, 0.7192) — all four top chunks from the same file
**Answer:** Before setting up a secure wireless network, plan your network infrastructure by identifying the number of users, devices, and access points required. Determine the type of encryption and authentication methods to be used. Consider the physical layout of your workspace and the range of your wireless signal.
**Sources:** 008_Setting_Up_a_Secure_Wireless_Network.txt

### Q5 (off-topic / not in documents)
**Question:** what is rag?
**Retrieved:** 009_Resetting_a_Jammed_Printer.txt (1.0832, 1.1154, 1.1355, 1.2043) — all above `DISTANCE_THRESHOLD`, and irrelevant to the question (this dataset contains no RAG-concept documents, only IT-support tickets)
**Answer:** I could not found this in your documents.
**Sources:** (none)