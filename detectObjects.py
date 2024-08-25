import ultralytics
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def detected_objects(filename:str):
    results = model.predict(source=filename, conf=0.25)

    categories = results[0].names

    dc = []
    for i in range(len(results[0])):
        cat = results[0].boxes[i].cls
        dc.append(categories[int(cat)])

    print(dc)
    return results, dc

