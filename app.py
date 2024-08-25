import streamlit as st
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

st.title('Extract Objects From Image')

uploaded_file = st.file_uploader('Upload an image')

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    with open('uploaded_file.png','wb') as file:
        file.write(uploaded_file.getvalue())

    # Detect objects in the uploaded image
    # results, dc = detectObjects.detected_objects('uploaded_file.png')
    results, dc = detected_objects('uploaded_file.png')

    st.write(dc)

    option = st.selectbox("Which object would you like to extract?", tuple(dc))
    # print(option)
    index_of_the_choosen_detected_object = tuple(dc).index(option)

    if st.button('Extract'):
        for result in results:
            boxes = result.boxes

        bbox=boxes.xyxy.tolist()[index_of_the_choosen_detected_object]
        # sam_checkpoint = "sam_vit_b_01ec64.pth"
        # model_type = "vit_b"
        # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        # predictor = SamPredictor(sam)

        image = cv2.cvtColor(cv2.imread('uploaded_file.png'), cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        input_box = np.array(bbox)

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        # plt.figure(figsize=(10, 10))
        # st.image(image)
        # plt.imshow(image)
        # show_mask(masks[0], plt.gca())
        # show_box(input_box, plt.gca())
        # plt.axis('off')
        # plt.show()

        segmentation_mask = masks[0]
        binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

        white_background = np.ones_like(image) * 255

        new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]


        plt.imsave('extracted_image.jpg', new_image.astype(np.uint8))
        # st.image('extracted_image.jpg')

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
        # st.image(output_path)

        with open("finalExtracted.png", "rb") as file:
            btn = st.download_button(
                label="Download final image",
                data=file,
                file_name="finalExtracted.png",
                mime="image/png",
            )

        # bbox=boxes.xyxy.tolist()[0]



