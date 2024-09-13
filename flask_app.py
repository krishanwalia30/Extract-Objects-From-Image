# import streamlit as st
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# from IPython.display import display, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
from PIL import Image


# Content of detectObjects.py file 
# import detectObjects
import ultralytics
from ultralytics import YOLO
import base64
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin

model = YOLO('yolov8n.pt')
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

def detected_objects(filename:str):
    results = model.predict(source=filename, conf=0.25)

    categories = results[0].names

    dc = []
    for i in range(len(results[0])):
        cat = results[0].boxes[i].cls
        dc.append(categories[int(cat)])

    print(dc)
    return results, dc

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


app = Flask(__name__)
CORS(app)

@app.route("/", methods = ['GET'])
@cross_origin()
def home():
    html_content = "<h1>SERVER UP!</h1>"
    return html_content, 200, {'Content-Type': 'text/html'}

@app.route('/detected-objects', methods=['POST'])
def extract_objects():
    """
        Request: Pass the image file to be processed. The file should be sent as form-data with the key 'image'

        Response: Returns a JSON with the detected objects and the base64 encoded image.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files.getlist('image')
    print(image_file)

    total_list = set()

    for image in image_file:
        image.save('uploaded_file.png')

        results, dc = detected_objects('uploaded_file.png')
        
        image_file = open('uploaded_file.png', 'rb')
        image_str = base64.b64encode(image_file.read())
        image_file.close()

        decodeit = open('hello_level.png', 'wb') 
        decodeit.write(base64.b64decode((image_str))) 
        decodeit.close() 


    return jsonify({"Objects_Detected": dc, "image_str": str(image_str)}), 200

@app.route("/extracted-object", methods=['POST'])
def extracted_image():
    """
        Request: Pass the index of the object to be extracted from the image already passed to the api endpoint '/detected-objects'

        Response: Returns a JSON with the base64 encoded image of the extracted object.
    """
    # if 'image' not in request.files:
    #     return jsonify({"error": "No image uploaded"}), 400
    
    # image_file = request.files.getlist('image')
    # index = request.get_json()

    # print(index)
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(image_file)



    # for image in image_file:
    #     image.save('uploaded_file.png')

    #     results, dc = detected_objects('uploaded_file.png')
        
    #     image_file = open('uploaded_file.png', 'rb')
    #     image_str = base64.b64encode(image_file.read())
    #     image_file.close()
# //////////////////////////////////////////////////////////////////// A new way below, for just parsing the index that has been passed in the request

    index_of_the_choosen_detected_object = request.get_json()['index']

    print(index_of_the_choosen_detected_object)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    results, dc = detected_objects('uploaded_file.png')
    for result in results:
        boxes = result.boxes

    bbox=boxes.xyxy.tolist()[index_of_the_choosen_detected_object]

    image = cv2.cvtColor(cv2.imread('uploaded_file.png'), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    input_box = np.array(bbox)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    segmentation_mask = masks[0]
    binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

    white_background = np.ones_like(image) * 255

    new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]

    plt.imsave('extracted_image.jpg', new_image.astype(np.uint8))

    # Store path of the image in the variable input_path
    input_path = 'extracted_image.jpg'

    # Store path of the output image in the variable output_path
    output_path = 'finalExtracted.png'

    # Processing the image
    input = Image.open(input_path)

    # Removing the background from the given Image
    output = remove(input)

    #Saving the image in the given path
    output.save(output_path)


    # Converting the final image to base64 string
    image_file = open(output_path, 'rb')
    image_str = base64.b64encode(image_file.read())
    image_file.close()


    return jsonify({'success': True, 'image_str': str(image_str)}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

