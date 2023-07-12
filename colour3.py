import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance


def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def adjust_saturation(image, saturation_factor):
    img_pil = Image.fromarray(image)
    enhancer = ImageEnhance.Color(img_pil)
    enhanced_image = enhancer.enhance(saturation_factor)
    return np.array(enhanced_image)


def adjust_sharpness(image, sharpness_factor):
    img_pil = Image.fromarray(image)
    enhancer = ImageEnhance.Sharpness(img_pil)
    enhanced_image = enhancer.enhance(sharpness_factor)
    return np.array(enhanced_image)


def perform_object_detection(image):
    # Load YOLO weights and configuration
 

    net = cv2.dnn.readNetFromDarknet("C:/Users/Shetakshi/OneDrive/Documents/GitHub/shetakshi_yadav/yolov3.cfg", "C:/Users/Shetakshi/OneDrive/Documents/GitHub/shetakshi_yadav/yolov3.weights")

    # Verify if the network was loaded successfully
    if net.empty():
        st.write("Failed to load YOLO network.")
        return image

    # Load the class labels
    with open("C:/Users/Shetakshi/OneDrive/Documents/GitHub/shetakshi_yadav/coco.names", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Get the output layer names
    layer_names = net.getLayerNames()
    output_layers = []
    # for i in net.getUnconnectedOutLayers():
    #   output_layers.append(layer_names[i[0] - 1])

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Resize the image for object detection
    input_size = (416, 416)
    resized_image = cv2.resize(image, input_size)

    # Normalize the image and convert it to a blob
    blob = cv2.dnn.blobFromImage(resized_image, 1 / 255.0, input_size, swapRB=True, crop=False)

    # Set the input for the network
    net.setInput(blob)

    # Perform forward pass and get the output layer outputs
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
                center_x, center_y, w, h = (detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and class labels
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        label = f'{classes[class_id]}: {confidences[i]:.2f}'

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters and Object Detection!")
    st.text("We use OpenCV, Streamlit, and YOLO for this demo")

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    saturation_factor = st.sidebar.slider("Saturation", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    sharpness_factor = st.sidebar.slider("Sharpness", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    enable_object_detection = st.sidebar.checkbox('Enable Object Detection')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    processed_image = blur_image(original_image, blur_rate)
    processed_image = brighten_image(processed_image, brightness_amount)
    processed_image = adjust_saturation(processed_image, saturation_factor)
    processed_image = adjust_sharpness(processed_image, sharpness_factor)

    if enable_object_detection:
        processed_image = perform_object_detection(processed_image)

    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image])


if __name__ == '__main__':
    main_loop()
