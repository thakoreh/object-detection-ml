from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from fastapi import FastAPI,Response,UploadFile,File
import requests
from fastapi.responses import FileResponse
from io import BytesIO
import base64
import re


app = FastAPI()

@app.post("/")
async def postImg(file: bytes = File()):
    if not file:
        return {"message": "No file sent"}
    else:
        objects = []
        image = Image.open(BytesIO(file))
        feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # let's only keep detections with score > 0.9
            if score > 0.9:
                objects.append(model.config.id2label[label.item()])
        return {"objects":objects}
                