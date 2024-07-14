from ts.torch_handler.base_handler import BaseHandler
import torch


class MyCustomHandler(BaseHandler):
    def __init__(self):
        super(MyCustomHandler, self).__init__()

    def preprocess(self, data):
        incoming_text = []
        for row in data:
            text = row.get("body")
            if text is None:
                raise ValueError("No image data found in request")
            
            print(f"preprocessor input: {text}")
            incoming_text.append(text)

        return torch.stack(incoming_text)

    def postprocess(self, data):
        return data
