import uvicorn
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
import tensorflow as tf
app = FastAPI()

import tensorflow as tf
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form
from typing import List
from fastapi.responses import HTMLResponse

from .utils import FeatureProcessor, load_image_into_numpy_array, construct_vw_example

the_feature_processor = FeatureProcessor.create(
        feature_processor_name="MobileNet",
        batch_size=32,
        feature_file_format=None
)

# class Reviews(BaseModel):
#     review: str

@app.post("/teach/")
async def create_teach(files: List[UploadFile] = File(...), labels: str = Form(...)):
    # Todo: move some of these to long running tasks
    individual_labels = labels.split(',')

    individual_image_features = []
    for the_file in files:
        print(f"working on {the_file.filename}")
        image = load_image_into_numpy_array(await the_file.read())
        print(f" loaded to numpy array! {image.shape}")

        image_features =\
                the_feature_processor.create_features_for_an_image(
                    the_feature_processor.process_in_memory_image(
                        image
                    )
                )
        # assert image_features.shape == (1, 62720), "Unexpected ImageNet shape!"
        
#         individual_image_features.append(
#             the_feature_processor.create_features(
#                 the_feature_processor.process_in_memory_image(
#                     image
#                 )
#         )
#     )
        print(f"... taught {the_file.filename}!")
    
#     a_key = get_or_assign_learner()
#     image_feature_ex = pool_of_learners[a_key].example(
#         construct_vw_example(TRUE, the_image_features)
#     )
#     pool_of_learners[a_key].learn(image_feature_ex)

#     decision = pool_of_learners[a_key].predict(image_feature_ex)

    
    return {
        "files": [file.filename for file in files],
        "labels": individual_labels
    }

@app.get("/")
async def main():
    content = """
<body>
<form action="/teach/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input name="labels" type="text" required>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)