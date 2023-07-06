import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def main():
    st.title("Object Recognition with VGG16")
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

        # Perform object recognition
        predictions = model.predict(img_processed)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Display the top predictions
        st.subheader("Top Predictions:")
        for pred in decoded_predictions:
            st.write(f"{pred[1]}: {pred[2]*100:.2f}%")

        # Detect lines and circles
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(img_gray_blur, 50, 150)

        # Detect lines using Hough line transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        # Draw lines on the image
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Detect circles using Hough circle transform
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

        # Draw circles on the image
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)

        # Display the image with predicted labels and detected patterns
        st.subheader("Image with Predicted Labels and Detected Patterns")
        st.image(img, use_column_width=True, channels="RGB")

if __name__ == "__main__":
    main()
