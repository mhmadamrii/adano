# Adaptive Personal Note Classification with User-Defined Categories Using Sentence Embeddings

**A Web-Based System Built with React.js and FastAPI**

> **Status: Hypothetical thesis.** This document is a template/proposal. All numbers in Chapter 6 are **illustrative placeholders, not measured results**. Replace them with real measurements after you run your own experiments.

---

## Abstract

Personal note-taking apps usually rely on manual folders or tags, which people stop maintaining over time. This thesis proposes a web-based system in which users define their own categories (e.g., "Todo", "Important News", "Ideas") and the system automatically assigns each new note to the most appropriate category. The system uses a pre-trained multilingual sentence-embedding model to convert notes into vectors, and a hybrid classifier that combines (1) similarity to the category description for cold start and (2) similarity to user-confirmed examples as data accumulates. A user correction loop turns every correction into new training data, so accuracy improves with use without retraining a large model. The backend is implemented in FastAPI, the frontend in React.js, and the whole system is designed to run on a small CPU-only VPS. The thesis describes the design, implementation plan, and evaluation protocol for measuring accuracy, cold-start behavior, learning speed, and latency.

**Keywords:** text classification, sentence embeddings, few-shot learning, user-defined categories, personal knowledge management, FastAPI, React

---

## Chapter 1: Introduction

### 1.1 Background

People capture short, unstructured text all day: reminders, ideas, links, opinions, news snippets. Examples:

- "I need to take money from ATM" → a task.
- "Bitcoin will go to 100K" → a market-related note the user may want flagged as important.

Organizing these manually costs effort, so most notes end up in one unsorted pile. Automatic classification can remove that effort, but a fixed set of categories does not fit everyone. One user's "Important" is another user's "Noise".

### 1.2 Problem Statement

1. Fixed-category classifiers cannot adapt to the categories each user wants.
2. Users create new categories with zero or very few examples, so classical supervised models (which need hundreds of labeled samples) fail at cold start.
3. Retraining a model each time a user adds a category or a correction is expensive and slow.

### 1.3 Research Questions

- **RQ1:** How accurate is embedding-based classification when users define categories with only a name and short description (zero examples)?
- **RQ2:** How quickly does accuracy improve as users confirm or correct predictions?
- **RQ3:** Can the system run with acceptable latency on a low-cost CPU-only VPS?

### 1.4 Objectives

1. Design a system where each user can create, rename, and delete categories at any time.
2. Build a classifier that works immediately (cold start) and improves with feedback.
3. Implement the system with React.js (frontend) and FastAPI (backend).
4. Evaluate accuracy, learning curve, and latency.

### 1.5 Scope and Limitations

**In scope:** short text notes (under ~500 characters), single-label classification, per-user category sets, English and Indonesian/mixed-language notes (via a multilingual model).

**Out of scope:** multi-label classification, images/audio, collaborative shared categories, large-scale multi-tenant optimization.

### 1.6 Significance

The approach shows a lightweight alternative to LLM-API classification: lower recurring cost, data stays on your own server, and no per-request external dependency.

---

## Chapter 2: Literature Review

> **Citation note:** The references below are from memory and have not been verified against a database. Check each one (authors, year, title) before using it in a real thesis.

### 2.1 Text Classification

Classic approaches use bag-of-words or TF-IDF features with linear models (Logistic Regression, SVM). They are fast but match keywords, not meaning: "grab cash from the machine" and "take money from ATM" share almost no words.

### 2.2 Pre-trained Language Models and Sentence Embeddings

Transformer models such as BERT (Devlin et al., 2019) produce contextual representations. Sentence-BERT (Reimers & Gurevych, 2019) adapts them to produce fixed-size sentence vectors where semantically similar sentences are close in cosine similarity. Small models such as MiniLM variants run on CPU in milliseconds per sentence.

### 2.3 Few-Shot and Zero-Shot Classification

- **Zero-shot:** compare the input embedding to an embedding of the label text/description.
- **Few-shot:** use a handful of labeled examples per class, via nearest-neighbor, prototype (centroid), or a lightweight classifier on embeddings.

### 2.4 Interactive / Human-in-the-Loop Learning

Systems that let users correct predictions and learn from the corrections reduce the cost of labeling, because labeling happens naturally during normal use.

### 2.5 Research Gap

Most work assumes a fixed label set known at training time. This thesis targets **dynamic, per-user label sets** where categories appear at runtime with little or no data.

---

## Chapter 3: System Design

### 3.1 Requirements

**Functional**

| ID  | Requirement                                                            |
| --- | ---------------------------------------------------------------------- |
| F1  | User can register/login                                                |
| F2  | User can create, edit, delete categories (name + optional description) |
| F3  | User can submit a note; system returns predicted category + confidence |
| F4  | Low-confidence notes go to "Uncategorized"                             |
| F5  | User can confirm or change the predicted category                      |
| F6  | System uses confirmations/corrections to improve future predictions    |
| F7  | User can browse, search, and filter notes by category                  |

**Non-functional**

| ID  | Requirement                                                   |
| --- | ------------------------------------------------------------- |
| N1  | Classification latency under 300 ms on a 2 GB RAM, 2 vCPU VPS |
| N2  | No GPU required                                               |
| N3  | Adding a category must not require retraining a large model   |
| N4  | User data is isolated per account                             |

### 3.2 Architecture

```
┌────────────┐   HTTPS    ┌──────────┐    ┌────────────────────────┐
│  React.js  │ ─────────► │  Nginx   │ ─► │  FastAPI (Uvicorn)     │
│  (SPA)     │ ◄───────── │ (reverse │    │  - Auth (JWT)          │
└────────────┘            │  proxy)  │    │  - Category CRUD       │
                          └──────────┘    │  - Classification svc  │
                                          │  - Feedback svc        │
                                          └──────────┬─────────────┘
                                                     │
                      ┌──────────────────────────────┼───────────────────┐
                      ▼                              ▼                   ▼
             ┌─────────────────┐          ┌───────────────────┐  ┌─────────────────┐
             │ Embedding model │          │ PostgreSQL        │  │ (optional)      │
             │ (loaded once in │          │ users, categories │  │ pgvector or     │
             │  memory)        │          │ notes, embeddings │  │ in-memory cache │
             └─────────────────┘          └───────────────────┘  └─────────────────┘
```

Key decision: **the embedding model is frozen and shared by all users.** Only lightweight per-user data (category descriptions, example embeddings) changes. This is why adding a category or a correction is instant.

### 3.3 Technology Stack

| Layer      | Choice                                                                          | Reason                            |
| ---------- | ------------------------------------------------------------------------------- | --------------------------------- |
| Frontend   | React.js (+ Vite)                                                               | Component model, fast dev         |
| Backend    | FastAPI                                                                         | Async, typed, auto-generated docs |
| ML         | `sentence-transformers`, NumPy, scikit-learn                                    | Embeddings and evaluation tools   |
| Model      | `paraphrase-multilingual-MiniLM-L12-v2` (or `all-MiniLM-L6-v2` if English only) | Small, CPU-friendly               |
| Database   | PostgreSQL (SQLite for prototype)                                               | Reliable, relational              |
| Deployment | Docker, Nginx, Linux VPS                                                        | Cheap, self-hosted                |

### 3.4 Database Schema

```sql
CREATE TABLE users (
  id            SERIAL PRIMARY KEY,
  email         TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE categories (
  id          SERIAL PRIMARY KEY,
  user_id     INT REFERENCES users(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  description TEXT,                -- optional, used for cold start
  desc_embedding BYTEA,            -- embedding of "name: description"
  UNIQUE (user_id, name)
);

CREATE TABLE notes (
  id                    SERIAL PRIMARY KEY,
  user_id               INT REFERENCES users(id) ON DELETE CASCADE,
  text                  TEXT NOT NULL,
  embedding             BYTEA NOT NULL,
  category_id           INT REFERENCES categories(id) ON DELETE SET NULL, -- final label
  predicted_category_id INT REFERENCES categories(id) ON DELETE SET NULL,
  confidence            REAL,
  is_confirmed          BOOLEAN DEFAULT FALSE, -- true if user confirmed or corrected
  created_at            TIMESTAMPTZ DEFAULT now()
);
```

Only notes with `is_confirmed = TRUE` are used as training examples. This prevents the model from reinforcing its own mistakes.

### 3.5 API Design

| Method | Endpoint                        | Purpose                                                       |
| ------ | ------------------------------- | ------------------------------------------------------------- |
| POST   | `/auth/register`, `/auth/login` | Authentication                                                |
| GET    | `/categories`                   | List user's categories                                        |
| POST   | `/categories`                   | Create category `{name, description?}`                        |
| PATCH  | `/categories/{id}`              | Edit category                                                 |
| DELETE | `/categories/{id}`              | Delete category                                               |
| POST   | `/notes`                        | Create note → returns prediction                              |
| GET    | `/notes?category_id=`           | List/filter notes                                             |
| PATCH  | `/notes/{id}/category`          | Confirm or correct category (feedback)                        |
| POST   | `/notes/reclassify`             | Re-run prediction on unconfirmed notes after category changes |

### 3.6 Classification Algorithm

Let `e(x)` be the normalized embedding of text `x`. For user category `c`:

- `d_c` = embedding of `"{name}: {description}"` (description-based prototype)
- `E_c` = set of embeddings of confirmed notes in category `c`, with `n_c = |E_c|`

**Score per category:**

```
sim_desc(x, c) = cos(e(x), d_c)
sim_ex(x, c)   = mean of top-k cos(e(x), v) for v in E_c      (k = min(3, n_c))

alpha_c = min(1, n_c / N0)                  # N0 = 10 (tunable)
score(x, c) = (1 - alpha_c) * sim_desc + alpha_c * sim_ex
```

- With 0 examples, the category is judged purely by its description (zero-shot).
- As examples accumulate, example similarity takes over.

**Confidence:** apply softmax with temperature `T` to the scores:

```
p_c = exp(score_c / T) / sum_j exp(score_j / T)
```

**Decision rule:**

- Predict `argmax p_c` if `p_max >= tau` **and** `top1 - top2 >= margin`.
- Otherwise assign to **Uncategorized** and ask the user.

`tau`, `margin`, `T`, and `N0` are tuned on a validation set (Chapter 5).

**Optional upgrade:** once a category set has enough confirmed data (e.g., 30+ per category), train a Logistic Regression on the embeddings and compare it with the hybrid scorer. Use whichever scores higher on validation.

### 3.7 Feedback Loop

1. User submits a note → system predicts → UI shows the category with confidence.
2. User accepts (one click) or changes the category.
3. Note is stored with `is_confirmed = TRUE` and its embedding joins `E_c`.
4. Next prediction automatically uses the updated `E_c`. No retraining step.

### 3.8 Frontend Design (React.js)

Main components:

- `NoteInput`: text box; on submit shows predicted category as a chip with confidence.
- `CategorySidebar`: list of categories with counts, create/edit/delete.
- `NoteList`: filterable list; each item has a category dropdown for correction.
- `UncategorizedInbox`: notes waiting for user decision.

State management: React Query (or SWR) for server state; local state for the input form. Optimistic updates for category corrections so the UI feels instant.

### 3.9 Edge Cases

| Case                                   | Handling                                                                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| User has no categories                 | Prompt to create at least one; skip classification                                                      |
| Category deleted                       | Its notes move to Uncategorized; embeddings removed from scoring                                        |
| Category renamed or description edited | Recompute `d_c`                                                                                         |
| Two near-duplicate categories          | Margin rule sends ambiguous notes to Uncategorized; UI can warn on high category-to-category similarity |
| Very short note ("ok", "hmm")          | Low confidence by design, goes to Uncategorized                                                         |
| Mixed language notes                   | Use the multilingual model                                                                              |

---

## Chapter 4: Implementation Plan

### 4.1 Backend Core (simplified)

```python
# classifier.py
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # loaded once

def embed(text: str) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True)

def softmax(x, T=0.05):
    z = np.array(x) / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()

def classify(text, categories, N0=10, k=3, tau=0.5, margin=0.1):
    """
    categories: list of dicts {id, desc_vec, example_vecs (np.ndarray [n, d])}
    returns (category_id | None, confidence)
    """
    if not categories:
        return None, 0.0

    x = embed(text)
    scores = []
    for c in categories:
        sim_desc = float(x @ c["desc_vec"])
        ex = c["example_vecs"]
        n = len(ex)
        if n > 0:
            sims = np.sort(ex @ x)[::-1][: min(k, n)]
            sim_ex = float(sims.mean())
            alpha = min(1.0, n / N0)
        else:
            sim_ex, alpha = 0.0, 0.0
        scores.append((1 - alpha) * sim_desc + alpha * sim_ex)

    probs = softmax(scores)
    order = np.argsort(probs)[::-1]
    top1 = probs[order[0]]
    top2 = probs[order[1]] if len(order) > 1 else 0.0

    if top1 >= tau and (top1 - top2) >= margin:
        return categories[order[0]]["id"], float(top1)
    return None, float(top1)
```

```python
# main.py (excerpt)
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class NoteIn(BaseModel):
    text: str

@app.post("/notes")
def create_note(payload: NoteIn, user=Depends(get_current_user), db=Depends(get_db)):
    cats = load_user_categories(db, user.id)        # with desc_vec + example_vecs
    cat_id, conf = classify(payload.text, cats)
    note = save_note(db, user.id, payload.text, predicted=cat_id, confidence=conf)
    return {"note_id": note.id, "category_id": cat_id, "confidence": conf}

class FeedbackIn(BaseModel):
    category_id: int

@app.patch("/notes/{note_id}/category")
def confirm_category(note_id: int, payload: FeedbackIn,
                     user=Depends(get_current_user), db=Depends(get_db)):
    set_confirmed_category(db, user.id, note_id, payload.category_id)
    return {"ok": True}
```

> This is a sketch. A production version needs input validation, authorization checks on every query, rate limiting, and caching of per-user category vectors.

### 4.2 Performance Considerations

- Load the embedding model **once at startup**, not per request.
- Cache each user's category matrices in memory (invalidate on change).
- Embedding a short note on CPU typically takes tens of milliseconds for MiniLM-size models; verify on your own VPS.
- For users with thousands of examples per category, keep a capped set (e.g., latest 200 per category) or use pgvector/FAISS.

### 4.3 Deployment

1. Dockerize backend (model downloaded at build time so startup is fast).
2. Build React app, serve static files via Nginx.
3. Nginx reverse-proxies `/api` to Uvicorn.
4. HTTPS with Let's Encrypt.
5. Minimum VPS: 2 vCPU, 2 GB RAM (the model plus runtime needs roughly 1 GB).

### 4.4 Security and Privacy

- Passwords hashed with bcrypt or argon2.
- JWT with short expiry plus refresh tokens.
- Every query filtered by `user_id`.
- Notes may contain sensitive data: encrypted disk, HTTPS only, and a clear data-deletion path.
- No third-party API calls for classification, so note content stays on the server.

---

## Chapter 5: Evaluation Methodology

### 5.1 Datasets

Because personal notes are private, there is no standard public dataset for this task. Plan:

1. **Self-collected set:** 300-600 notes written by the researcher and volunteers, labeled into 5-8 categories (Todo, Important News, Idea, Personal, Finance, etc.).
2. **Multiple simulated users:** each user defines different categories to test the "user-defined" claim.
3. **Public proxy datasets (optional):** short-text classification datasets for sanity checks.

Label rules must be written down before labeling. Where two annotators disagree, record inter-annotator agreement (Cohen's kappa).

### 5.2 Experiments

| Exp. | Question                  | Setup                                                                                     |
| ---- | ------------------------- | ----------------------------------------------------------------------------------------- |
| E1   | Cold-start accuracy (RQ1) | Categories with name only vs. name + description, 0 examples                              |
| E2   | Learning curve (RQ2)      | Add confirmed examples 0, 1, 3, 5, 10, 20, 50 per category; measure accuracy at each step |
| E3   | Baselines                 | Compare against TF-IDF + LR, embeddings + LR, and an LLM-API prompt (if budget allows)    |
| E4   | Threshold study           | Vary `tau` and `margin`; plot accuracy vs. fraction auto-classified                       |
| E5   | Latency and memory (RQ3)  | Measure p50/p95 latency and RAM on the target VPS                                         |
| E6   | Category drift            | Add a new category mid-stream; measure impact on existing categories                      |
| E7   | Usability (optional)      | Small user study: time saved and satisfaction (SUS questionnaire)                         |

### 5.3 Metrics

- Accuracy and **macro-F1** (important when categories are imbalanced)
- Per-category precision/recall, confusion matrix
- **Coverage:** share of notes auto-classified (not sent to Uncategorized)
- **Accuracy at coverage X%:** accuracy on auto-classified notes only
- Latency (p50/p95), memory usage
- **Corrections per 100 notes** over time (a practical measure of how annoying the system is)

### 5.4 Validity Threats

- Small, self-collected dataset: results may not generalize.
- Researcher-written notes are cleaner than real notes. Prefer real notes collected over time.
- Category overlap (e.g., "Todo" vs. "Important") limits achievable accuracy for any model.
- Embedding model bias toward its training data; mixed-language and slang notes may perform worse.

---

## Chapter 6: Hypothetical Results and Discussion

> **Warning: the numbers below are made-up illustrations of what the results section could look like. They are not real measurements. Do not present them as findings.**

### 6.1 Illustrative Cold-Start Results (E1)

| Category info provided   | Accuracy (illustrative) |
| ------------------------ | ----------------------- |
| Name only                | ~55-65%                 |
| Name + short description | ~65-75%                 |

_Expected pattern:_ descriptions help because the name alone ("Todo") is a weak signal compared with a description ("things I need to do or buy, errands, reminders").

### 6.2 Illustrative Learning Curve (E2)

| Confirmed examples per category | Accuracy (illustrative) |
| ------------------------------- | ----------------------- |
| 0                               | ~70%                    |
| 5                               | ~78%                    |
| 10                              | ~83%                    |
| 20                              | ~87%                    |
| 50                              | ~90%                    |

_Expected pattern:_ quick gains in the first 10-20 examples, then diminishing returns.

### 6.3 Illustrative Baseline Comparison (E3)

| Method                | Expected relative behavior                                        |
| --------------------- | ----------------------------------------------------------------- |
| TF-IDF + LR           | Weak with under 50 examples per category                          |
| Embeddings + LR       | Good once each category has ~20+ examples; weak at cold start     |
| **Hybrid (proposed)** | Best at low data because it works from zero examples              |
| LLM API               | Strong at zero-shot, but per-request cost and external dependency |

### 6.4 Discussion (what the hypothesis predicts)

1. The hybrid scorer should outperform pure supervised methods in the first days of use, because it does not wait for training data.
2. Confidence thresholding trades coverage for accuracy. Users may prefer a few extra "Uncategorized" notes over wrong auto-labels.
3. Ambiguous categories remain the main error source, and no model fixes unclear definitions. The UI should warn when two categories are semantically very close.
4. The example "Bitcoin will go to 100K" shows the labeling issue: it is a prediction/opinion, not news. Whether it lands in "Important News" depends on how the user defines the category. The system learns the user's definition, not an objective truth.

---

## Chapter 7: Conclusion and Future Work

### 7.1 Conclusion

This thesis proposes a lightweight, self-hostable system for classifying personal notes into categories that users define themselves. By freezing a pre-trained multilingual embedding model and learning only from category descriptions and user-confirmed examples, the system supports instant category creation, works at cold start, and improves through everyday corrections. The React and FastAPI stack runs on a small CPU-only VPS. The proposed evaluation will measure whether the approach reaches practically useful accuracy with few labeled examples.

### 7.2 Limitations

- Single-label only.
- Accuracy depends on how clear and distinct the user's categories are.
- Short or context-dependent notes ("call him tomorrow") are inherently ambiguous.
- Embedding quality is bounded by the chosen pre-trained model.

### 7.3 Future Work

- Multi-label classification (a note can be both "Todo" and "Finance").
- Category suggestions: cluster Uncategorized notes and propose new categories.
- Active learning: ask for labels on the notes the model is least sure about.
- Optional fine-tuning of the embedding model on aggregated, consented data.
- Support for images, voice notes, and links (extract text, then classify).
- On-device inference (ONNX/WebAssembly) for fully offline privacy.

---

## References

> Unverified, from memory. Check before citing.

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding._ NAACL-HLT.
2. Reimers, N., & Gurevych, I. (2019). _Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks._ EMNLP-IJCNLP.
3. Pedregosa, F., et al. (2011). _Scikit-learn: Machine Learning in Python._ Journal of Machine Learning Research, 12.
4. Wang, W., et al. (2020). _MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers._ NeurIPS.
5. FastAPI documentation: https://fastapi.tiangolo.com
6. React documentation: https://react.dev
7. Sentence-Transformers documentation: https://www.sbert.net

---

## Appendix A: Example Category Definitions

| Category       | Description (used for cold start)                                 |
| -------------- | ----------------------------------------------------------------- |
| Todo           | Things I need to do, buy, pay, or remember; errands and reminders |
| Important News | News, market updates, or announcements I want to keep track of    |
| Ideas          | Project ideas, things to try, creative thoughts                   |
| Personal       | Feelings, journal-style thoughts, relationships                   |
| Finance        | Budgeting, expenses, investments, money-related notes             |

## Appendix B: Suggested Project Structure

```
project/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── auth.py
│   │   ├── models.py          # SQLAlchemy models
│   │   ├── schemas.py         # Pydantic schemas
│   │   ├── classifier.py      # embedding + scoring logic
│   │   └── routers/
│   │       ├── categories.py
│   │       └── notes.py
│   ├── tests/
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/        # NoteInput, CategorySidebar, NoteList
│   │   ├── api/
│   │   └── App.jsx
│   └── package.json
├── evaluation/
│   ├── dataset.csv
│   └── run_experiments.py
└── docker-compose.yml
```
