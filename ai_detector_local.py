import os
import re
import json
import torch
import pickle
import hashlib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel
from pycparser import CParser

CACHE_DIR = ".embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class AIDetectorLocal:
    def __init__(self, model_path=None, dataset_path='code_dataset.json', device=None, use_cache=True):
        self.model = None
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.threshold = 0.45

        self.dataset_path = dataset_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cache = use_cache

        # Load CodeBERT
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.codebert.eval()

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.train_model()

    # ------------------------------
    # Feature Extraction
    # ------------------------------
    def _extract_handcrafted_features(self, code):
        code_no_comments = re.sub(r'//.*|/\*.*?\*/', '', code, flags=re.DOTALL)
        comments = re.findall(r'//.*|/\*.*?\*/', code, flags=re.DOTALL)
        comment_ratio = len(' '.join(comments)) / max(len(code), 1)
        depth, max_depth = 0, 0
        for c in code:
            if c == '{': depth += 1; max_depth = max(max_depth, depth)
            elif c == '}': depth -= 1
        lines = [len(l) for l in code.splitlines() if l.strip()]
        avg_line = np.mean(lines) if lines else 0
        identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', code)
        uniq_ratio = len(set(identifiers)) / max(len(identifiers), 1)
        return np.array([comment_ratio, max_depth, avg_line, uniq_ratio])

    def _extract_ast_features(self, code):
        parser = CParser()
        try:
            ast = parser.parse(code)
        except:
            return np.zeros(4)
        ast_str = str(ast)
        node_count = ast_str.count('(')
        control_count = sum(ast_str.count(k) for k in ['If', 'For', 'While', 'Switch'])
        func_defs = ast_str.count('FuncDef')
        avg_node_density = len(ast_str) / max(node_count, 1)
        return np.array([node_count, control_count, func_defs, avg_node_density])

    def _get_code_hash(self, code):
        return hashlib.sha256(code.encode('utf-8')).hexdigest()

    def _load_cached_embedding(self, code_hash):
        path = os.path.join(CACHE_DIR, f"{code_hash}.npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def _save_cached_embedding(self, code_hash, emb):
        path = os.path.join(CACHE_DIR, f"{code_hash}.npy")
        np.save(path, emb)

    @torch.no_grad()
    def _extract_codebert_embedding(self, code):
        code_hash = self._get_code_hash(code)
        if self.use_cache:
            cached = self._load_cached_embedding(code_hash)
            if cached is not None:
                return cached
        tokens = self.tokenizer(code, return_tensors='pt', truncation=True, max_length=256)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.codebert(**tokens)
        emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        if self.use_cache:
            self._save_cached_embedding(code_hash, emb)
        return emb

    # ------------------------------
    # Training
    # ------------------------------
    def train_model(self):
        try:
            with open(self.dataset_path, 'r') as f:
                data = json.load(f)
            codes = [item['code'] for item in data]
            labels = [item['label'] for item in data]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.initialize_default_model()
            return

        if not codes:
            print("Dataset empty — using fallback model.")
            self.initialize_default_model()
            return

        X_train, X_test, y_train, y_test = train_test_split(codes, labels, test_size=0.2, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 3))
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        X_train_hand = np.vstack([self._extract_handcrafted_features(c) for c in X_train])
        X_test_hand = np.vstack([self._extract_handcrafted_features(c) for c in X_test])
        X_train_ast = np.vstack([self._extract_ast_features(c) for c in X_train])
        X_test_ast = np.vstack([self._extract_ast_features(c) for c in X_test])
        X_train_cb = np.vstack([self._extract_codebert_embedding(c) for c in X_train])
        X_test_cb = np.vstack([self._extract_codebert_embedding(c) for c in X_test])

        self.scaler.fit(np.hstack([X_train_hand, X_train_ast]))
        X_train_combined = np.hstack([
            X_train_tfidf.toarray(),
            self.scaler.transform(np.hstack([X_train_hand, X_train_ast])),
            X_train_cb
        ])
        X_test_combined = np.hstack([
            X_test_tfidf.toarray(),
            self.scaler.transform(np.hstack([X_test_hand, X_test_ast])),
            X_test_cb
        ])

        self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
        self.model.fit(X_train_combined, y_train)

        preds = self.model.predict(X_test_combined)
        acc = accuracy_score(y_test, preds)
        print(f"✅ Hybrid local model accuracy: {acc:.2f}")
        print(classification_report(y_test, preds))

        probs = self.model.predict_proba(X_test_combined)
        ai_probs = [p[1] for i, p in enumerate(probs) if y_test[i] == 1]
        if ai_probs:
            self.threshold = max(0.5, np.mean(ai_probs) - np.std(ai_probs) / 2)

    def initialize_default_model(self):
        self.vectorizer = TfidfVectorizer(max_features=300)
        self.model = GradientBoostingClassifier()
        dummy_codes = [
            "int add(int a,int b){return a+b;}",
            "int factorial(int n){if(n<=1)return 1;return n*factorial(n-1);}"
        ]
        y = [0, 1]
        X_vec = self.vectorizer.fit_transform(dummy_codes)
        feats = np.vstack([self._extract_handcrafted_features(c) for c in dummy_codes])
        self.scaler.fit(feats)
        X_combined = np.hstack([X_vec.toarray(), self.scaler.transform(feats)])
        self.model.fit(X_combined, y)

    # ------------------------------
    # Prediction
    # ------------------------------
    def predict(self, code):
        try:
            tfidf_vec = self.vectorizer.transform([code])
            hand_feat = self._extract_handcrafted_features(code)
            ast_feat = self._extract_ast_features(code)
            cb_emb = self._extract_codebert_embedding(code)
            combined = np.hstack([
                tfidf_vec.toarray(),
                self.scaler.transform([np.hstack([hand_feat, ast_feat])]),
                cb_emb.reshape(1, -1)
            ])
            prob = self.model.predict_proba(combined)[0][1]
            return float(prob), prob >= self.threshold
        except Exception as e:
            print(f"Error predicting: {e}")
            return 0.5, False

    # ------------------------------
    # Save/Load
    # ------------------------------
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'scaler': self.scaler,
                'threshold': self.threshold
            }, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.scaler = data['scaler']
        self.threshold = data.get('threshold', 0.7)