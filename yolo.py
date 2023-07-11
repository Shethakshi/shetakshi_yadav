import streamlit as st
import cv2
import numpy as np

# Load the YOLO model
net = cv2.dnn.readNet("C:/Users/Shetakshi/Downloads/yolov3.weights", "C:/Users/Shetakshi/Downloads/yolov3.cfg")

# Load the class labels
with open("C:/Users/Shetakshi/Downloads/coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]


# Define a function for object recognition
def perform_object_recognition(image):
    # Resize the image to the input size of the YOLO model
    height, width, _ = image.shape
    input_size = (416, 416)
    resized_image = cv2.resize(image, input_size)

    # Normalize the image and convert it to a blob
    blob = cv2.dnn.blobFromImage(resized_image, 1 / 255.0, input_size, swapRB=True, crop=False)

    # Set the input for the network
    net.setInput(blob)

    # Forward pass through the network
    outputs = net.forward(output_layers)

    # Process the outputs
    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections by confidence threshold
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and class labels
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        label = f'{classes[class_id]}: {confidences[i]:.2f}'

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


# Streamlit app code
def main():
    st.title("Object Recognition App")

    # File upload section
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Perform object recognition on the image
        output_image = perform_object_recognition(image)

        # Display the output image
        st.image(output_image, channels="BGR")


# Run the Streamlit app
if __name__ == "__main__":
    main()
