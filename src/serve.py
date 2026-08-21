from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """Tai model.joblib tu Amazon S3 khi server khoi dong."""

    # Dam bao thu muc models ton tai
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Tao S3 client
    s3 = boto3.client("s3")

    # Tai model
    s3.download_file(
        ARTIFACT_BUCKET,
        MODEL_KEY,
        MODEL_PATH,
    )

    print(
        f"Model downloaded: "
        f"s3://{ARTIFACT_BUCKET}/{MODEL_KEY} "
        f"-> {MODEL_PATH}"
    )


# Tai model khi server khoi dong
download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Dau vao:
    {
        "features": [f1, f2, ..., f10]
    }

    Dau ra:
    {
        "prediction": 0 | 1,
        "label": "thu_nhap_thap" | "thu_nhap_cao"
    }
    """

    if len(req.features) != 10:
        raise HTTPException(
            status_code=400,
            detail="Expected 10 features (adult income)",
        )

    pred = int(model.predict([req.features])[0])

    label = (
        "thu_nhap_cao"
        if pred == 1
        else "thu_nhap_thap"
    )

    return {
        "prediction": pred,
        "label": label,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )