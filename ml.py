import os
import re
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# -----------------------------
# Paths & Config
# -----------------------------
MODEL_PKL     = Path("heart_disease_model.pkl")
FEATURES_JSON = Path("heart_features.json")
CSV_PATH      = Path("heart_cleaned.csv")
MODEL_VERSION = "logreg_v1"

KB_DIR  = Path("kb")
RAG_DIR = Path("rag_index")
RAG_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)

RAG_MODEL_NAME    = os.getenv("RAG_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RAG_TOP_K         = int(os.getenv("RAG_TOP_K", "4"))
FORCE_REBUILD_RAG = os.getenv("FORCE_REBUILD_RAG", "0") == "1"

# -----------------------------
# FAISS or sklearn fallback
# -----------------------------
_USE_FAISS = True
try:
    import faiss
except Exception:
    _USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Train or load model
# -----------------------------
def train_or_load_model():
    if MODEL_PKL.exists() and FEATURES_JSON.exists():
        model = joblib.load(MODEL_PKL)
        features = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
        return model, features

    if not CSV_PATH.exists():
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PKL)
    FEATURES_JSON.write_text(json.dumps(list(X.columns), ensure_ascii=False, indent=2), encoding="utf-8")
    return model, list(X.columns)

model, FEATURES = train_or_load_model()

# -----------------------------
# Rule-based triage
# -----------------------------
TRIAGE_BANNERS = {
    "critical": "🚨 Doctor Meetup – Immediate",
    "high":     "🟠 Doctor Appointment – Soon",
    "medium":   "🟡 Monitor Closely",
    "low":      "🟢 Routine Follow-up",
}

TRIAGE_RECS = {
    "critical": [
        "Same-day clinical review / ER if chest pain now.",
        "ECG + troponin as indicated.",
        "Medication adherence check.",
        "Avoid strenuous activity until cleared.",
    ],
    "high": [
        "GP/cardiology appointment within 1–2 weeks.",
        "Optimize BP, lipids, glucose.",
        "Consider stress testing per clinician.",
        "Start/adjust lifestyle plan immediately.",
    ],
    "medium": [
        "Primary-care review in 4–8 weeks.",
        "Labs: lipids, HbA1c.",
        "Lifestyle changes (DASH diet, ~150 min/wk activity).",
        "Begin a home BP log.",
    ],
    "low": [
        "Maintain lifestyle (diet, activity).",
        "Annual check-ups.",
        "Monitor BP/lipids per routine.",
    ],
}

TYPICAL_ANGINA_SET = {0, 1}

def _get_num(d, key, cast=float, default=0):
    try:
        return cast(d.get(key, default))
    except Exception:
        return default

def _is_thal_defect(thal_val):
    try:
        t = int(thal_val)
    except Exception:
        return False
    return t in (6, 7)

def rule_risk_tier(data: dict) -> str:
    age       = _get_num(data, "age", int, 0)
    cp        = _get_num(data, "cp", int, 0)
    trestbps  = _get_num(data, "trestbps", int, 0)
    chol      = _get_num(data, "chol", int, 0)
    fbs       = _get_num(data, "fbs", int, 0)
    thalach   = _get_num(data, "thalach", int, 0)
    exang     = _get_num(data, "exang", int, 0)
    oldpeak   = _get_num(data, "oldpeak", float, 0.0)
    slope     = _get_num(data, "slope", int, 1)
    ca        = _get_num(data, "ca", int, 0)
    thal      = _get_num(data, "thal", int, 0)

    if (
        oldpeak >= 2.0
        or (exang == 1 and oldpeak >= 1.5)
        or _is_thal_defect(thal)
        or ca >= 2
        or trestbps >= 180
        or (int(cp) in TYPICAL_ANGINA_SET)
        or thalach < 100
    ):
        return "critical"

    if (
        (1.0 <= oldpeak <= 1.9)
        or (160 <= trestbps <= 179)
        or (chol >= 240)
        or (age >= 60)
        or (exang == 1)
        or (100 <= thalach <= 119)
        or (slope == 0)
    ):
        return "high"

    if (
        (130 <= trestbps <= 159)
        or (200 <= chol <= 239)
        or (45 <= age <= 59)
        or (120 <= thalach <= 139)
        or (fbs == 1)
    ):
        return "medium"

    return "low"

# -----------------------------
# RAG engine
# -----------------------------
def _chunk_text(text: str, max_words: int = 400):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur, count = [], [], 0
    for s in parts:
        w = len(s.split())
        if count + w > max_words and cur:
            chunks.append(" ".join(cur))
            cur, count = [s], w
        else:
            cur.append(s); count += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks

class LocalRAG:
    def __init__(self, kb_dir: Path, idx_dir: Path, model_name: str):
        self.kb_dir = kb_dir
        self.idx_dir = idx_dir
        self.model_name = model_name
        self.docs = []
        self.embs = None
        self.index = None
        self._load_or_build()

    def _scan_kb(self):
        return list(self.kb_dir.glob("*.txt")) + list(self.kb_dir.glob("*.md"))

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(self.model_name)

    def _emb(self, texts):
        return self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def _faiss_write(self, embs: np.ndarray):
        faiss.write_index(self.index, str(self.idx_dir / "faiss.index"))
        with open(self.idx_dir / "docs.jsonl", "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _faiss_read(self):
        self.index = faiss.read_index(str(self.idx_dir / "faiss.index"))
        self.docs = []
        with open(self.idx_dir / "docs.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.docs.append(json.loads(line))

    def _sk_write(self, embs: np.ndarray):
        np.save(self.idx_dir / "embs.npy", embs)
        (self.idx_dir / "docs.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in self.docs),
            encoding="utf-8"
        )

    def _sk_read(self):
        self.embs = np.load(self.idx_dir / "embs.npy")
        self.docs = []
        with open(self.idx_dir / "docs.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.docs.append(json.loads(line))

    def _needs_build(self):
        if FORCE_REBUILD_RAG:
            return True
        if _USE_FAISS:
            return not (self.idx_dir / "faiss.index").exists() or not (self.idx_dir / "docs.jsonl").exists()
        else:
            return not (self.idx_dir / "embs.npy").exists() or not (self.idx_dir / "docs.jsonl").exists()

    def _build_index(self):
        files = self._scan_kb()
        self._load_model()
        texts = []
        for fp in files:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
            for chunk in _chunk_text(raw, max_words=400):
                texts.append({"text": chunk, "source": fp.name})

        if not texts:
            texts = [{"text": "No knowledge base provided.", "source": "README"}]

        self.docs = texts
        embs = self._emb([d["text"] for d in self.docs])

        if _USE_FAISS:
            dim = embs.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embs)
            self._faiss_write(embs)
        else:
            self.embs = embs
            self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
            self.nn.fit(self.embs)
            self._sk_write(embs)

    def _load_or_build(self):
        self.idx_dir.mkdir(exist_ok=True)
        if self._needs_build():
            self._build_index()
        else:
            if _USE_FAISS:
                self._faiss_read()
                self._load_model()
            else:
                self._sk_read()
                self._load_model()

    def rebuild(self):
        self._build_index()

    def retrieve(self, query: str, k: int = 4):
        k = max(1, min(k, 10))
        q = self._emb([query])
        if _USE_FAISS:
            scores, idxs = self.index.search(q, k)
            out = []
            for i, s in zip(idxs[0], scores[0]):
                if int(i) < 0: continue
                d = self.docs[int(i)]
                out.append({"score": float(s), "text": d["text"], "source": d["source"]})
            return out
        else:
            dists, idxs = self.nn.kneighbors(q, n_neighbors=k, return_distance=True)
            out = []
            for i, dist in zip(idxs[0], dists[0]):
                sim = 1.0 - float(dist)
                d = self.docs[int(i)]
                out.append({"score": sim, "text": d["text"], "source": d["source"]})
            return out

rag = LocalRAG(KB_DIR, RAG_DIR, RAG_MODEL_NAME)

def make_rag_query(features: dict, tier: str) -> str:
    parts = [f"{k}={features.get(k)}" for k in FEATURES]
    return f"Triage tier={tier}. " + ", ".join(parts)

def templated_summary(tier: str, features: dict, passages: list) -> dict:
    headline = {
        "critical": "Immediate medical attention is recommended.",
        "high":     "Prompt clinical evaluation is recommended.",
        "medium":   "Follow-up with primary care is advised.",
        "low":      "Routine preventive care is appropriate."
    }[tier]

    preview_bullets = []
    for p in passages[:3]:
        text = p["text"].strip()
        if len(text) > 300:
            text = text[:300].rstrip() + "..."
        preview_bullets.append(f"{text} ({p['source']})")

    summ = (
        f"{headline} Inputs: age={features.get('age')}, trestbps={features.get('trestbps')}, "
        f"chol={features.get('chol')}, thalach={features.get('thalach')}, "
        f"oldpeak={features.get('oldpeak')}, exang={features.get('exang')}."
    )
    return {
        "summary": summ,
        "highlights": preview_bullets,
        "citations": [{"source": p["source"], "score": round(p["score"], 3)} for p in passages]
    }

# -----------------------------
# Flask API
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

@app.route("/")
def index():
    return jsonify({"message": "Heart Disease Prediction API with RAG is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": "Missing feature(s).", "missing": missing}), 400

        input_df = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
        proba = float(model.predict_proba(input_df)[0][1])

        tier = rule_risk_tier(data)
        triage_banner = TRIAGE_BANNERS[tier]
        recommendations = TRIAGE_RECS[tier]

        query = make_rag_query(data, tier)
        passages = rag.retrieve(query, k=RAG_TOP_K)
        rag_payload = templated_summary(tier, data, passages)

        return jsonify({
            "risk_tier": tier,
            "risk_label": tier.capitalize(),
            "triage": triage_banner,
            "recommendations": recommendations,
            "model_probability": round(proba, 4),
            "meta": {
                "model_version": MODEL_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            },
            "rag": rag_payload
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print("Server running...")
    app.run(debug=True, host="0.0.0.0", port=port)
