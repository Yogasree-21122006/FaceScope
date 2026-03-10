from fastapi import FastAPI, File, UploadFile
import torch
import pickle
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from model_helper import FaceEmbeddingNet
import io
import os

app = FastAPI()

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

# Get current folder path (important for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------
# Load Face Embedding Model
# -----------------------------
def load_model():
    model = FaceEmbeddingNet(embedding_dim=128)

    model_path = os.path.join(BASE_DIR, "face_embedding_model_cpu.pth")

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# -----------------------------
# Load Stored Embeddings
# -----------------------------
embedding_path = os.path.join(BASE_DIR, "embeddings.pkl")

with open(embedding_path, "rb") as f:
    embedding_db = pickle.load(f)


# -----------------------------
# Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])


# -----------------------------
# Cosine Similarity Function
# -----------------------------
def cosine_similarity(a, b):

    if a.dim() == 1:
        a = a.unsqueeze(0)

    if b.dim() == 1:
        b = b.unsqueeze(0)

    similarity = F.cosine_similarity(a, b, dim=1)

    return similarity.item()


# -----------------------------
# Health Check (optional but useful)
# -----------------------------
@app.get("/")
def home():
    return {"message": "FaceScope API Running"}


# -----------------------------
# Face Verification API
# -----------------------------
@app.post("/verify")
async def verify_face(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        test_embedding = model(img_tensor)
        test_embedding = test_embedding.squeeze(0)

    best_match = None
    best_score = -1

    for name, db_embedding in embedding_db.items():

        db_embedding = torch.tensor(
            db_embedding,
            dtype=torch.float32
        ).to(DEVICE)

        score = cosine_similarity(test_embedding, db_embedding)

        if score > best_score:
            best_score = score
            best_match = name

    result = "VERIFIED" if best_score > THRESHOLD else "UNKNOWN"

    return {
        "best_match": best_match,
        "similarity_score": float(best_score),
        "result": result
    }
