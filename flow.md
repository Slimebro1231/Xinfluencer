# Bot‑Influencer Architecture & Data Flywheel

This document captures the full technical plan for a self‑learning, personality‑driven AI agent that monitors Twitter and Telegram for RWA‑gold & crypto chatter, filters noise, learns continuously, and publishes posts/comments under its own brand voice.  
It combines a **data‑flywheel** with Retrieval‑Augmented Generation (RAG), daily LoRA micro‑tuning, and a behaviour policy trained with reinforcement learning from engagement signals.

---

## 🔄 High‑Level Flow Diagram

```mermaid
flowchart TD
    %% ====  SOURCES  ====
    A(Twitter<br/>Stream API) -->|raw JSON| FQ(Data&nbsp;Quality Gate)
    B(Telegram<br/>Public Chats) -->|raw messages| FQ
    C(Internal<br/>Knowledge KG) --> IDX
    %% ====  QUALITY GATE  ====
    FQ -->|clean<br/>chunks| EMB(Embed<br/>→ vectors)
    FQ -->|drop spam<br/>toxicity| TRASH{{Discard}}

    %% ====  RETRIEVAL LOOP ====
    EMB -->|HNSW / IVF‑PQ| IDX[Vector DB<br/>(Pinecone/Qdrant)]
    IDX -->|top‑k IDs| RERANK(Cross‑encoder<br/>re‑rank)

    %% ====  GENERATION LOOP ====
    RERANK -->|context| GEN(LLM +<br/>LoRA adapters)
    GEN -->|draft+critique| SELF(Self‑RAG & Reflexion)
    SELF -->|persona‑consistent<br/>reply| POST(Post to<br/>Twitter/Telegram)

    %% ====  BEHAVIOUR LOOP ====
    POST -->|engagement<br/>(likes RT clicks)| REWARD(Reward Store)
    REWARD -->|weekly PPO| POLICY(Behaviour Policy)
    POLICY -->|reading &<br/>posting decisions| A
    POLICY -->|reading &<br/>posting decisions| B

    %% ====  METRICS & OPS ====
    IDX -. metrics .-> MON[Dashboard<br/>(RAGAS/Prom)]
    GEN -. faithfulness .-> MON
    POST -. brand drift .-> MON
```

---

## 1. Data‑Quality Gate (Loop 1)

| Check | Method | Threshold | Action |
|-------|--------|-----------|--------|
| Language | `fastText` lang‑ID | non‑English? | filter |
| Toxicity | Google Perspective | > 0.80 | discard |
| Bot score | Botometer‑Lite | top 10 % | discard |
| Perplexity band | GPT‑2 PPL | keep 10‑90 % | keep |
| Engagement | likes + RT above median | ✓ | priority |

Only messages passing **all** checks are chunked (256 tokens) and embedded.

---

## 2. Retrieval Loop (Loop 2)

* **Embedding model:** `bge‑large‑en` (or `text‑embedding‑3‑small` if using OpenAI).  
* **Index:** HNSW < 10 M vectors; migrate to IVF‑PQ + GPU search above that.  
* **Re‑ranking:** ColBERT‑v2 cross‑encoder adds ~10–15 % precision.  
* **Key metrics:** precision@5, recall@10, context‑precision (RAGAS).  
* **Trigger:** re‑index when precision@5 drops by 5 % WoW.

---

## 3. Generation Loop (Loop 3)

| Hyper‑param | Value | Note |
|-------------|-------|------|
| Base model | Mistral 7B or Llama‑3 8B | open‑weights |
| LoRA rank `r` | 16 | quality / VRAM trade‑off |
| Alpha | 2 × r | scaling rule |
| LR (AdamW) | 1 × 10⁻⁴ | tune first |
| Epochs | 1 | avoid over‑fit |

Daily micro‑adapters are merged back every 4–6 weeks to prevent adapter sprawl.

**Self‑RAG + Reflexion**: model critiques and iterates once before final post; cuts hallucinations ~30 %.

---

## 4. Behaviour Loop (Loop 4)

* **Reward model inputs:** likes, retweets, CTR, follower delta (positive); toxicity, off‑topic, low faithfulness (negative).  
* **Policy learner:** PPO updated weekly with 1‑step importance sampling.  
* **Safety valve:** if faithfulness < 0.9 or toxicity > 0.6, auto‑block posting & alert human.

---

## 5. Daily Cadence (SGT)

| Time | Job | Loop |
|------|-----|------|
| 01:00 | Hydrate 24 h sources → run quality gate | 1 |
| 02:00 | Embed & upsert vectors; rebuild ANN & re‑rank index | 2 |
| 03:00 | LoRA fine‑tune on new high‑quality chunks | 3 |
| 09 – 23 h | Agent reads, answers, posts (Self‑RAG active) | 2‑3 |
| 23:30 | Aggregate metrics → reward logs → PPO update | 4 |

---

## 6. Golden‑Signal Dashboard

| Signal | Threshold | Action |
|--------|-----------|--------|
| Retrieval precision@5 | ↓ > 5 % WoW | re‑index |
| Faithfulness | < 0.90 | tighten filters / retrain generator |
| Hallucination rate | ↑ WoW | increase Self‑RAG passes |
| p95 latency | > 2 s | scale index / lower k |
| Adapter count | > 8 | merge weights |

---

## 7. Knowledge‑Graph Hybrid

Immutable data (e.g., LBMA rules) lives in a Neo4j KG.  
Retriever first queries KG; if miss, fall back to vector DB, ensuring critical facts never drift.

---

## 8. Future Enhancements

* **Multi‑lingual switch**: add language‑specific adapters & embeddings.  
* **On‑device summariser**: distil daily streams into trend reports.  
* **Synthetic user simulator**: generate interaction traces to pre‑train behaviour policy before launch.

---

