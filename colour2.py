import cv2
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


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


def detect_objects_vgg(image):
    # Load the pre-trained VGG16 model
    model = VGG16(weights='imagenet')

    # Resize the image to fit the VGG16 input shape
    resized_img = cv2.resize(image, (224, 224))

    # Preprocess the image for VGG16
    img_processed = preprocess_input(resized_img)

    # Perform object recognition
    predictions = model.predict(np.expand_dims(img_processed, axis=0))
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the top predictions
    st.subheader("Top Predictions:")
    for pred in decoded_predictions:
        st.write(f"{pred[1]}: {pred[2]*100:.2f}%")

    return image


def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters, Object Detection, and Recognition!")
    st.text("We use OpenCV, Streamlit, and VGG16 for this demo")

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    saturation_factor = st.sidebar.slider("Saturation", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    sharpness_factor = st.sidebar.slider("Sharpness", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
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

    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)

    if enable_object_detection:
        processed_image = detect_objects_vgg(processed_image)

    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image])


if __name__ == '__main__':
    main_loop()


