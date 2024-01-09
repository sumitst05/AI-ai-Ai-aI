import cv2
import numpy as np


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = resize_image(image)
    return resized_image


def resize_image(image):
    image = np.array(image)
    image = cv2.resize(image, (28, 28))
    image = cv2.bitwise_not(image)
    return image


def predicting(image, model):
    image = image.reshape(1, 28 * 28)
    score_array = model.predict(image)
    prediction = np.argmax(score_array)
    return prediction
