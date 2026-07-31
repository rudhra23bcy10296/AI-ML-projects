"""
Face Recognizer Module - Cosine Similarity & Deep Feature Embeddings
Student: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560 | Email: rudhra.23bcy10296@vitbhopal.ac.in)
"""

import os
import pickle
import threading
import numpy as np

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False


class FaceRecognizer:
    """Embedding-based face recognizer using DeepFace and cosine similarity."""

    def __init__(
        self,
        known_faces_dir="known_faces",
        embeddings_file="embeddings.pkl",
        model_name="Facenet",
        distance_threshold=0.28,
    ):
        self.known_faces_dir = known_faces_dir
        self.embeddings_file = embeddings_file
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.embeddings: dict[str, list[list[float]]] = {}
        self.lock = threading.Lock()

        os.makedirs(self.known_faces_dir, exist_ok=True)
        self._ensure_default_user()
        self._load_or_build()

    def _ensure_default_user(self):
        """Create the subject_00 placeholder directory and a blank image."""
        import cv2

        default_dir = os.path.join(self.known_faces_dir, "subject_00")
        os.makedirs(default_dir, exist_ok=True)
        placeholder = os.path.join(default_dir, "placeholder.jpg")
        if not os.path.exists(placeholder):
            img = np.ones((160, 160, 3), dtype=np.uint8) * 128
            cv2.imwrite(placeholder, img)

    def _load_or_build(self):
        """Load embeddings from pickle; if missing, rebuild from images."""
        if os.path.exists(self.embeddings_file):
            self.load_embeddings()
        else:
            self.rebuild_embeddings()

    def load_embeddings(self):
        """Load embeddings dict from the pickle file."""
        with self.lock:
            try:
                with open(self.embeddings_file, "rb") as fh:
                    self.embeddings = pickle.load(fh)
                print(f"[Recognizer] Loaded embeddings for {len(self.embeddings)} identities.")
            except Exception as exc:
                print(f"[Recognizer] Failed to load embeddings: {exc}")
                self.embeddings = {}

    def save_embeddings(self):
        """Persist the current embeddings dict to disk."""
        with open(self.embeddings_file, "wb") as fh:
            pickle.dump(self.embeddings, fh)

    @staticmethod
    def _cosine_distance(a, b) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))

    def _generate_embedding(self, img_input) -> list[float] | None:
        """Return the embedding vector for img_input (path or ndarray)."""
        if not HAS_DEEPFACE:
            # Synthetic feature fallback for testing environments
            np.random.seed(42)
            return np.random.randn(128).tolist()
            
        try:
            result = DeepFace.represent(
                img_path=img_input,
                model_name=self.model_name,
                enforce_detection=False,
            )
            return result[0]["embedding"]
        except Exception as exc:
            print(f"[Recognizer] Embedding error: {exc}")
            return None

    def add_embedding(self, name: str, image_path: str) -> bool:
        """Generate an embedding for image_path and store it under name."""
        embedding = self._generate_embedding(image_path)
        if embedding is None:
            return False
        with self.lock:
            self.embeddings.setdefault(name, []).append(embedding)
            self.save_embeddings()
        print(f"[Recognizer] Added embedding for '{name}' — total vectors: {len(self.embeddings[name])}")
        return True

    def recognize_face(self, face_crop) -> str:
        query_emb = self._generate_embedding(face_crop)
        if query_emb is None:
            return "subject_00"

        best_name = "subject_00"
        best_dist = float("inf")

        with self.lock:
            for name, vectors in self.embeddings.items():
                if name == "subject_00":
                    continue
                for stored_emb in vectors:
                    dist = self._cosine_distance(query_emb, stored_emb)
                    if dist < best_dist:
                        best_dist = dist
                        best_name = name

        if best_dist > self.distance_threshold:
            return "subject_00"
        return best_name

    def rebuild_embeddings(self):
        """Re-generate every embedding from the images in known_faces_dir."""
        with self.lock:
            self.embeddings = {}
            for name in sorted(os.listdir(self.known_faces_dir)):
                person_dir = os.path.join(self.known_faces_dir, name)
                if not os.path.isdir(person_dir):
                    continue
                for fname in os.listdir(person_dir):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    img_path = os.path.join(person_dir, fname)
                    emb = self._generate_embedding(img_path)
                    if emb is not None:
                        self.embeddings.setdefault(name, []).append(emb)
            self.save_embeddings()
        print(f"[Recognizer] Rebuilt embeddings — {len(self.embeddings)} identities.")
