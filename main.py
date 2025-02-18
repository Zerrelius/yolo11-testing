from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Perform object detection on an image
results = model(["./images/image4.jpg"])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    #masks = result.masks  # Masks object for segmentation masks outputs
    #keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    #obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="./results/result.jpg")  # save to disk
