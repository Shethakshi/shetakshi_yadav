import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def main():
    st.title("Object Detection with VGG16")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the pre-trained VGG16 model
        model = VGG16(weights='imagenet')

        # Read the uploaded image
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess the image
        img_processed = cv2.resize(img, (224, 224))
        img_processed = image.img_to_array(img_processed)
        img_processed = np.expand_dims(img_processed, axis=0)
        img_processed = preprocess_input(img_processed)

        # Perform object detection
        predictions = model.predict(img_processed)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Display the top predictions
        st.subheader("Top Predictions:")
        for pred in decoded_predictions:
            st.write(f"{pred[1]}: {pred[2]*100:.2f}%")

        # Detect objects and draw bounding boxes
        img_with_boxes = detect_objects(img, decoded_predictions)

        # Display the image with predicted labels and bounding boxes
        st.subheader("Image with Predicted Labels and Bounding Boxes")
        st.image(img_with_boxes, use_column_width=True, channels="RGB")

def detect_objects(img, predictions):
    # Create a copy of the image to draw bounding boxes on
    img_with_boxes = img.copy()

    # Get the image dimensions
    height, width, _ = img.shape

    # Iterate over the predictions and draw bounding boxes
    for pred in predictions:
        label, _, confidence = pred
        label = f"{label}: {confidence*100:.2f}%"

        # Get the predicted class index
        class_index = np.argmax(pred)

        # Get the bounding box coordinates
        x, y, w, h = get_bounding_box(class_index, width, height)

        # Draw the bounding box on the image
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_with_boxes, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img_with_boxes

def get_bounding_box(class_index, width, height):
    # Define the bounding box coordinates for each class
    bounding_boxes = {
        0: (50, 50, 100, 100),  # Example bounding box for class 0
        1: (100, 100, 150, 150),  # Example bounding box for class 1
        2: (150, 150, 200, 200)  # Example bounding box for class 2
        # Add more bounding boxes for other classes if needed
    }

    # Get the bounding box coordinates for the given class index
    if class_index in bounding_boxes:
        return bounding_boxes[class_index]
    else:
        # If class index not found, return a default bounding box
        return (0, 0, width, height)

if __name__ == "__main__":
    main()




