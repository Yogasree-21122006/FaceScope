from fastapi import FastAPI, File, UploadFile
import torch
import pickle
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from model_helper import FaceEmbeddingNet
import io

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5


def load_model():
    model = FaceEmbeddingNet(embedding_dim=128)
    model.load_state_dict(
        torch.load("face_embedding_model_cpu.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

with open("embeddings.pkl", "rb") as f:
    embedding_db = pickle.load(f)


transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])


def cosine_similarity(a, b):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    similarity = F.cosine_similarity(a, b, dim=1)
    return similarity.item()


@app.get("/")
def home():
    return {"message": "Face Verification API Running"}


@app.post("/verify")
async def verify_face(file: UploadFile = File(...)):
    try:

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

    except Exception as e:
        return {"error": str(e)}
