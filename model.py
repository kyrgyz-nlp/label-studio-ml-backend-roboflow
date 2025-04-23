from label_studio_ml.model import LabelStudioMLBase
from inference_sdk import InferenceHTTPClient
import requests
from PIL import Image
from io import BytesIO
import os

class RoboflowModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(RoboflowModel, self).__init__(**kwargs)
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set.")
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        self.model_id = "newspaper-only-articles/4"

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            image_url = task['data'].get('image')
            if not image_url:
                continue

            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                result = self.client.infer(image, model_id=self.model_id)
            except Exception as e:
                print(f"Error processing image: {e}")
                continue

            annotations = []
            for pred in result.get('predictions', []):
                x = pred['x']
                y = pred['y']
                width = pred['width']
                height = pred['height']
                annotations.append({
                    'from_name': 'label',
                    'to_name': 'image',
                    'type': 'rectanglelabels',
                    'value': {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height,
                        'rectanglelabels': [pred['class']]
                    }
                })

            predictions.append({'result': annotations})
        return predictions
