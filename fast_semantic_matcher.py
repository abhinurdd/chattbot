# fast_semantic_matcher.py
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional

# Lazy global model to reduce cold start time
_sentence_model = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class FastSemanticMatcher:
    """
    AI + embeddings-based influencer matcher with FAISS support.
    - Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) for text embeddings.
    - Uses FAISS IndexFlatIP with L2-normalized vectors so inner product == cosine.
    """
    def __init__(self):
        self.influencer_data: Dict[str, Dict] = {}
        self.is_loaded: bool = False
        self.faiss_index: Optional[faiss.Index] = None
        self.username_list: List[str] = []
        self.embedding_dim: int = 384  # all-MiniLM-L6-v2 outputs 384 dims [6]

    # ---------- Embeddings ----------
    def _ensure_model(self):
        global _sentence_model
        if _sentence_model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
            _sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # [6]
        return _sentence_model

    def get_text_embedding(self, text: str) -> List[float]:
        """Return a normalized 384-d vector for a single text."""
        model = self._ensure_model()
        vec = model.encode([text], normalize_embeddings=True)  # shape (1, 384) L2-normalized [8]
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)    # shape (384,)
        return vec.tolist()

    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Batch encode texts to a float32 NxD matrix, L2-normalized."""
        model = self._ensure_model()
        vectors = model.encode(texts, normalize_embeddings=True)  # shape (N, 384) [8]
        return np.asarray(vectors, dtype=np.float32)

    # ---------- Load precomputed influencer summaries ----------
    def load_precomputed_embeddings(self):
        """
        Load influencer_embeddings.json (summaries/metadata).
        Try to load FAISS index + username mapping if present.
        """
        try:
            print("⚡ Loading precomputed influencer summaries...")
            with open('influencer_embeddings.json', 'r') as f:
                embeddings_data = json.load(f)

            self.influencer_data = {}
            for username, data in embeddings_data.items():
                self.influencer_data[username] = data['influencer_data']

            self.is_loaded = True
            print(f"✅ Loaded {len(self.influencer_data)} influencer summaries")

            # Load FAISS index (optional)
            try:
                self.load_faiss_index('influencer_index.faiss', 'username_mapping.json')
            except Exception as e:
                print(f"ℹ️ FAISS index not loaded yet ({e}). You can create it with generate_embeddings.py")

        except FileNotFoundError:
            print("❌ influencer_embeddings.json not found. Run generate_embeddings.py first.")
            self.is_loaded = False
        except Exception as e:
            print(f"❌ Error loading summaries: {e}")
            self.is_loaded = False

    # ---------- FAISS index helpers ----------
    def build_faiss_index(self, embeddings: np.ndarray, usernames: List[str]):
        """
        Build an IndexFlatIP (cosine via inner product on normalized vectors).
        embeddings: float32 array of shape (N, D)
        usernames: list of length N
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Sanity checks
        assert embeddings.ndim == 2, f"Expected 2D embeddings, got shape={embeddings.shape}"
        assert embeddings.shape > 0 and embeddings.shape[18] > 0, f"Empty embeddings matrix: shape={embeddings.shape}"
        assert len(usernames) == embeddings.shape, "usernames length must match embeddings rows"

        # Normalize and set dimension
        faiss.normalize_L2(embeddings)  # defensively normalize [1]
        d = embeddings.shape[18]         # correct feature dimension (N, D) -> D [5]
        index = faiss.IndexFlatIP(d)    # inner product on normalized vectors == cosine [3]
        index.add(embeddings)

        self.faiss_index = index
        self.username_list = list(usernames)
        self.embedding_dim = d
        print(f"✅ FAISS index built with {len(usernames)} vectors (dim={d})")

    def save_faiss_index(self, index_path: str, mapping_path: str):
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built")
        faiss.write_index(self.faiss_index, index_path)
        with open(mapping_path, 'w') as f:
            json.dump({i: u for i, u in enumerate(self.username_list)}, f, indent=2)
        print(f"💾 Saved FAISS index to {index_path} and mapping to {mapping_path}")

    def load_faiss_index(self, index_path: str, mapping_path: str):
        self.faiss_index = faiss.read_index(index_path)
        with open(mapping_path, 'r') as f:
            id_to_username = json.load(f)
        # Ensure deterministic order by integer keys
        if isinstance(id_to_username, dict):
            max_i = max(int(k) for k in id_to_username.keys())
            self.username_list = [id_to_username[str(i)] for i in range(max_i + 1)]
        else:
            self.username_list = list(id_to_username)
        print(f"✅ Loaded FAISS index and mapping ({len(self.username_list)} usernames)")

    def search_by_text(self, text: str, top_k: int = 8) -> List[Tuple[str, float]]:
        """Return [(username, score)] by cosine similarity against FAISS index."""
        if self.faiss_index is None or not self.username_list:
            print("ℹ️ No FAISS index loaded. Run generate_embeddings.py to create it.")
            return []

        # Build (1, D) float32 query matrix
        q_vec = np.asarray(self.get_text_embedding(text), dtype=np.float32)  # (D,)
        if q_vec.ndim != 1:
            q_vec = q_vec.reshape(-1)
        q = np.expand_dims(q_vec, axis=0)  # (1, D)

        # Search: returns (1, top_k) arrays
        scores, idxs = self.faiss_index.search(q, top_k)
        
        # FIX: Access the first row of the 2D arrays
        scores_flat = scores[0]  # Convert (1, top_k) to (top_k,)
        idxs_flat = idxs[0]      # Convert (1, top_k) to (top_k,)
        
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores_flat, idxs_flat):
            if idx == -1:  # FAISS returns -1 for invalid/missing results
                continue
            if idx >= len(self.username_list):  # Bounds check
                continue
            username = self.username_list[idx]
            results.append((username, float(score)))
        return results


    # ---------- Matching path ----------
    def find_semantic_matches(self, product_description: str, top_k: int = 8) -> List[Dict]:
        """
        Use FAISS vector search if available; otherwise fall back to AI/keyword logic.
        Returns a list of influencer dicts with semantic_match_score, match_confidence, and rank.
        """
        if not self.is_loaded:
            self.load_precomputed_embeddings()

        # Vector search path
        if self.faiss_index is not None and self.username_list:
            hits = self.search_by_text(product_description, top_k=top_k)
            matches: List[Dict] = []
            for rank, (username, score) in enumerate(hits, start=1):
                if username in self.influencer_data:
                    influencer = self.influencer_data[username].copy()
                    influencer['semantic_match_score'] = float(score)
                    # For normalized vectors, IP in [0, 1]; convert to percentage confidence
                    influencer['match_confidence'] = max(0.0, min(1.0, float(score))) * 100.0
                    influencer['rank'] = rank
                    matches.append(influencer)
            return matches

        # Fallback matching (no FAISS index present)
        return self._fallback_keyword_matching(product_description, list(self.influencer_data.values()))

    def _fallback_keyword_matching(self, product_description: str, influencers: List[Dict]) -> List[Dict]:
        """Simple keyword-based scorer as final fallback."""
        product_keywords = (product_description or "").lower().split()
        matches: List[Dict] = []

        for influencer in influencers:
            score = 0
            category = (influencer.get('category') or '').lower()
            if category:
                for kw in product_keywords:
                    if kw in category:
                        score += 30

            hashtags = [str(tag).lower() for tag in influencer.get('top_hashtags', [])]
            for kw in product_keywords:
                if any(kw in h for h in hashtags):
                    score += 15

            bio = (influencer.get('bio') or '').lower()
            if bio:
                for kw in product_keywords:
                    if kw in bio:
                        score += 10

            if score > 0:
                infc = influencer.copy()
                infc['semantic_match_score'] = score / 100.0
                infc['match_confidence'] = min(score, 100)
                matches.append(infc)

        matches.sort(key=lambda x: x.get('semantic_match_score', 0), reverse=True)
        return matches[:8]

    def explain_match(self, influencer: Dict, product: str) -> str:
        if 'ai_reasoning' in influencer:
            return influencer['ai_reasoning']
        name = influencer.get('name', 'Unknown')
        category = influencer.get('category', 'Unknown')
        return f"Strong alignment: {name}'s {category.lower()} content fits {product}."


# Global instance
fast_semantic_matcher = FastSemanticMatcher()
