import io
import logging
import time

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile
)

from PIL import Image

from src.predict import Predictor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cats vs Dogs Classification API",
    version="1.0.0"
)

predictor = Predictor()

request_count = 0
total_latency = 0.0


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/metrics")
def metrics():

    average_latency = (
        total_latency / request_count
        if request_count > 0
        else 0.0
    )

    return {
        "request_count": request_count,
        "average_latency_ms": round(
            average_latency * 1000,
            2
        )
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    global request_count
    global total_latency

    start = time.perf_counter()

    try:

        if not file.content_type:
            raise HTTPException(
                status_code=400,
                detail="Missing content type"
            )

        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )

        result = predictor.predict(image)

        latency = time.perf_counter() - start

        request_count += 1
        total_latency += latency

        logger.info(
            "POST /predict "
            "prediction=%s "
            "latency_ms=%.2f",
            result["class"],
            latency * 1000
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )