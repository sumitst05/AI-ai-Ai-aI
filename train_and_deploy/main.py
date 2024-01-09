from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from pred_img import *

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

model = tf.keras.models.load_model("./saved_models")


@app.get("/ping")
async def ping():
    return "Server Up and running"


def read_file_as_image(image):
    byte_image = BytesIO(image)
    image = Image.open(byte_image)
    image = np.array(image)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File()):
    image = read_file_as_image(await file.read())
    # print(image)
    # print(image.shape)
    resized_image = preprocess(image)
    prediction = predicting(resized_image, model)
    print(prediction)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
