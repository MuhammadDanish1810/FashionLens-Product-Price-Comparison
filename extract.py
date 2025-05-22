from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from PIL import Image
import cv2

feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def enhance_image(img_array):
    img_array = img_array.astype(np.float32)
    if img_array.max() <= 1.0:
        img_array *= 255
    
    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    
    lab = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

def load_and_preprocess_image(img_input):
    try:
        if isinstance(img_input, np.ndarray):
            if img_input.ndim == 4:
                img_input = np.squeeze(img_input, axis=0)
            if img_input.ndim == 3 and img_input.shape[0] == 1:
                img_input = np.squeeze(img_input, axis=0)
            
            if img_input.ndim == 2 or (img_input.ndim == 3 and img_input.shape[-1] == 1):
                img_input = cv2.cvtColor(img_input.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            if img_input.shape[-1] != 3:
                raise ValueError("Expected image with 3 color channels (RGB)")
            
            img = Image.fromarray(img_input.astype(np.uint8))
        elif isinstance(img_input, str):
            img = Image.open(img_input).convert('RGB')
        else:
            raise TypeError("Input should be a numpy array or file path")

        img = img.resize((224, 224), Image.LANCZOS)
        
        img_array = np.array(img)
        img_array = enhance_image(img_array)
        
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"Error in load_and_preprocess_image: {str(e)}")
        raise

def extract_features(img_input):
    try:
        img_array = load_and_preprocess_image(img_input)
        features = feature_extractor.predict(img_array)
        
        features_flat = features.flatten()
        features_normalized = normalize(features_flat.reshape(1, -1))[0]
        
        return features_normalized
        
    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        raise

def compute_similarity(features1, features2):
    try:
        cosine_sim = cosine_similarity([features1], [features2])[0][0]
        
        min_similarity_threshold = 0.3 
        if cosine_sim < min_similarity_threshold:
            return 0.0
        
        scaled_similarity = (cosine_sim - min_similarity_threshold) / (1 - min_similarity_threshold)
        
        return max(0.0, min(1.0, scaled_similarity)) 
        
    except Exception as e:
        print(f"Error in compute_similarity: {str(e)}")
        return 0.0
