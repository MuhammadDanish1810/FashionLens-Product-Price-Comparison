import streamlit as st
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import requests
from extract import *
from scrapping import *
from io import BytesIO
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def load_custom_model():
    model = tf.keras.models.load_model("cnn_model.keras")
    return model

def load_metadata():
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    return metadata.get("unique_product_types", [])

def preprocess_image_for_classification(image):
    img = np.array(image.convert('L'))
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(-1, 28, 28, 1)
    return img

def predict_with_custom_model(model, unique_product_types, img):
    processed_img = preprocess_image_for_classification(img)
    prediction = model.predict(processed_img, verbose=0)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    if 0 <= predicted_class_index < len(unique_product_types):
        return unique_product_types[predicted_class_index], max(prediction[0])
    else:
        return "Unknown", 0.0

# Pretrained ResNet-50 model functions
def load_pretrained_model():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    return processor, model

def preprocess_image_for_pretrained(image, processor):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def predict_label_with_pretrained(processor, model, image):
    inputs = preprocess_image_for_pretrained(image, processor)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()
    
    predicted_label = model.config.id2label[predicted_class_idx]
    return predicted_label, confidence

# Fetch image from URL
def fetch_image_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            st.error(f"Failed to fetch image from URL. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching image from URL: {str(e)}")
        return None

# Main function
def main():
    st.title("Price Comparison Application")
    
    # Load models
    try:
        custom_model = load_custom_model()
        unique_product_types = load_metadata()
        pretrained_processor, pretrained_model = load_pretrained_model()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    # Input method selection
    input_method = st.radio("Choose input method:", ("Upload Image", "Enter Image URL"))

    image = None
    predicted_label = None
    confidence = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            with st.spinner('Analyzing image...'):
                # Use custom model for classification
                predicted_label, confidence = predict_with_custom_model(custom_model, unique_product_types, image)
    else:
        url = st.text_input("Enter the URL of an image:")
        if url:
            image = fetch_image_from_url(url)
            if image:
                with st.spinner('Analyzing image...'):
                    # Use pretrained model for URL images
                    predicted_label, confidence = predict_label_with_pretrained(pretrained_processor, pretrained_model, image)

    if image is not None and predicted_label is not None:
        st.image(image, caption="Input Image", use_container_width=True)
        
        # Display prediction
        st.subheader("Prediction:")
        st.write(f" Predicted Item: {predicted_label.title()}")
        st.write(f" Confidence: {confidence:.2%}")

        # Scraping Settings in the Sidebar
        st.sidebar.subheader("Scraping Settings")
        max_products = st.sidebar.slider("Number of images to scrape per website:", 1, 100, 20)
        similarity_threshold = st.sidebar.slider("Similarity threshold:", 0.0, 1.0, 0.2)

        # Similarity search
        if st.button("Search Similar Products"):
            with st.spinner('Searching for similar products...'):
                try:
                    # Extract features for similarity search
                    img_for_features = np.array(image.convert('RGB'))
                    uploaded_features = extract_features(img_for_features)
                    
                    # Scrape products from Amazon and eBay
                    scraped_products = []
                    scraped_products.extend(scrape_amazon_products(predicted_label, max_products=max_products))
                    scraped_products.extend(scrape_ebay_products(predicted_label, max_products=max_products))

                    if not scraped_products:
                        st.error("No products found on any platform.")
                    else:
                        products_with_similarity = []

                        for product in scraped_products:
                            try:
                                image_urls = product.get("image_urls", [])
                                best_similarity = 0
                                best_image = None

                                for image_url in image_urls:
                                    response = requests.get(image_url)
                                    if response.status_code != 200:
                                        continue
                                        
                                    scraped_img = Image.open(BytesIO(response.content))
                                    scraped_img_array = np.array(scraped_img.convert('RGB'))
                                    scraped_features = extract_features(scraped_img_array)
                                    similarity_score = compute_similarity(uploaded_features, scraped_features)

                                    if similarity_score > best_similarity:
                                        best_similarity = similarity_score
                                        best_image = scraped_img

                                if best_similarity > similarity_threshold:
                                    products_with_similarity.append({
                                        "image": best_image,
                                        "similarity": best_similarity,
                                        "price": product["price"],
                                        "link": product["link"],
                                        "title": product["title"],
                                        "source": product["source"]
                                    })

                            except Exception as e:
                                continue

                        products_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)

                        if products_with_similarity:
                            st.subheader("Most Similar Product:")
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.image(products_with_similarity[0]["image"], width=200)
                            with col2:
                                st.write(f"**Title:** {products_with_similarity[0]['title']}")
                                st.write(f"**Price:** {products_with_similarity[0]['price']}")
                                st.write(f"**Source:** {products_with_similarity[0]['source']}")
                                st.write(f"**Similarity Score:** {products_with_similarity[0]['similarity']:.2f}")
                                st.write(f"**Product Link:** [View on {products_with_similarity[0]['source']}]({products_with_similarity[0]['link']})")

                            if len(products_with_similarity) > 1:
                                st.subheader("Other Similar Products:")
                                for product in products_with_similarity[1:]:
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.image(product["image"], width=150)
                                    with col2:
                                        st.write(f"**Title:** {product['title']}")
                                        st.write(f"**Price:** {product['price']}")
                                        st.write(f"**Source:** {product['source']}")
                                        st.write(f"**Similarity Score:** {product['similarity']:.2f}")
                                        st.write(f"**Product Link:** [View on {product['source']}]({product['link']})")
                                    st.divider()
                        else:
                            st.warning("No products met the similarity threshold.")
                        
                except Exception as e:
                    st.error(f"Error during similarity search: {str(e)}")

if __name__ == "__main__":
    main()