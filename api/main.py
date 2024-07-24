from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAME = ["EARLY BLIGHT", "LATE BLIGHT", "HEALTHY"]

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")

async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    #bytes = await file.read()

    image = read_file_as_image(await file.read())

    image_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(image_batch)

    # json_data = {
    #     "instances": image_batch.tolist()
    # }
    #
    # requests.post(endpoint, json=json_data)

    predicted_class = CLASS_NAME[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)