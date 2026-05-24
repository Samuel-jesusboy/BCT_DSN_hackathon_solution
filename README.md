# DSN × BCT LLM Agent Challenge — Submission

> **Task A: User Modeling** | **Task B: Recommendation Agent**
> Generator: Qwen3-4B | Embedding: BAAI/bge-small-en-v1.5 | Dataset: Yelp (6.9M reviews)

---

## Final Results

### Task A — User Modeling (4-Way Ablation)

| Experiment | RMSE ↓ | ROUGE-1 ↑ | BERTScore ↑ | Fidelity |
|---|---|---|---|---|
| **Prompted + Neutral** | **1.2470** | **0.1995** | **0.8492** | 0.5% |
| Prompted + Nigerian | 1.2470 | 0.1964 | 0.8419 | **80.0%** |
| LoRA + Neutral | 2.1448 | 0.1629 | 0.8402 | 2.0% |
| LoRA + Nigerian | 2.3022 | 0.1698 | 0.8365 | 6.0% |

### Task B — Recommendation Agent

| Metric | Warm User | Cold-Start |
|---|---|---|
| NDCG@10 | 0.7101 | 0.5051 |
| Hit Rate@10 | 0.8700 | 0.8850 |
| Rating Alignment | 0.7617 | — |

---

## Project Structure

```
├── app/
│   └── main.py                  # FastAPI app — all endpoints
├── artifacts/                   # Pre-built FAISS index + user history
│   ├── item_index.faiss
│   ├── user_history.pkl
│   ├── item_profiles.pkl
│   ├── item_context.pkl
│   └── persona_embeddings.npy
├── notebooks/
│   └── datascience-final-solution.ipynb   # Full training notebook (Kaggle)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option 1 — Docker (recommended)
```bash
git clone https://github.com/YOUR_USERNAME/dsn-bct-agent.git
cd dsn-bct-agent
export HF_TOKEN=your_hf_token_here
docker-compose up --build
# API live at http://localhost:8000
```

### Option 2 — Local Python
```bash
pip install -r requirements.txt
python app/main.py
```

---

## API Endpoints

### Task A — Generate Review
```bash
POST /generate-review
{
  "user_id": "mh_-eMZ6K5RLWhZyISBhwA",
  "item_id":  "XQfwVwDr-v0ZS3_CbbE5Xw",
  "approach": "nigerian"   # or "neutral"
}
```
**Response:**
```json
{
  "user_id": "mh_-eMZ6K5RLWhZyISBhwA",
  "item_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
  "predicted_rating": 4.0,
  "generated_review": "Chai! This place is really something else..."
}
```

### Task B — Recommendations
```bash
POST /recommend
{
  "user_id":    "mh_-eMZ6K5RLWhZyISBhwA",
  "context":    "good restaurant for a first date",
  "top_n":      5,
  "cold_start": false
}
```

### Task B — Multi-turn Chat
```bash
POST /chat
{
  "user_id":  "mh_-eMZ6K5RLWhZyISBhwA",
  "message":  "I need somewhere good to eat",
  "nigerian": true,
  "conversation_history": []
}
```

---

## Architecture

```
Yelp Reviews (6.9M)
        ↓
  Persona Encoder (bge-small-en-v1.5)
        ↓
  ┌─────────────────────┬────────────────────────┐
  Task A                Task B
  Two-step Generation   Retrieve → Rerank
  ├─ Rating: greedy     ├─ FAISS top-20
  │  (thinking OFF)     └─ Qwen3 CoT rerank
  └─ Review: sampled        (thinking ON)
     (thinking OFF)
```

**Key design decisions:**
- **Thinking mode asymmetry:** disabled for Task A (speed), enabled for Task B (reasoning quality)
- **Nigerian contextualisation:** prompt engineering only — no fine-tuning required
- **Cold-start:** direct query embedding into item FAISS index — no user history needed
- **4-bit quantization:** Qwen3-4B fits in 8GB VRAM via BitsAndBytes NF4

---

## Reproducing Results

1. Open `notebooks/datascience-final-solution.ipynb` on Kaggle
2. Add dataset: `dzakyrezandi/yelp-review-dataset`
3. Enable GPU T4 x2, set SEED=42
4. Run all cells (~3-4 hours)
5. Download `/kaggle/working/artifacts/` → place in `./artifacts/`

---

## Environment
- Python 3.10 | PyTorch 2.4 | Transformers ≥4.51.0
- Kaggle T4 x2 GPU (training) | SEED=42
"# BCT_DSN_hackathon_solution" 
