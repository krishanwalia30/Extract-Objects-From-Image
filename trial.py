# import requests

# API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
# headers = {"Authorization": "Bearer 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"}

# def query(filename):
#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.post(API_URL, headers=headers, data=data)
#     return response.json()

# output = query("uploaded_file.png")
# print(output)
# # with open('triallllllll.png', "wb") as f:
#     # f.write(output)

import requests
API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

query(
        payload={
            "image": "A HTTP POST request is used to ",
            "parameters": {"temperature": 0.8, "max_new_tokens": 50, "seed": 42},
        }
    )

