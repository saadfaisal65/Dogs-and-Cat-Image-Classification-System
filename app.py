import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
import numpy as np
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐾",
    layout="wide"
)

# Add some styling
st.markdown("""
<style>
    .prediction {
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .dog {
        background-color: #d4edda;
        color: #155724;
    }
    .cat {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .uncertain {
        background-color: #fff3cd;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced image preprocessing
def process_image(image):
    try:
        # Convert to PIL Image for better preprocessing
        img = Image.open(image).convert('RGB')
        
        # Get dimensions
        width, height = img.size
        
        # Determine cropping area (center crop)
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        
        # Perform center crop
        img = img.crop((left, top, right, bottom))
        
        # Resize to required dimensions
        img = img.resize((224, 224), Image.LANCZOS)
        
        # Convert to array
        img_array = np.array(img)
        
        # Data augmentation for test images can improve accuracy
        # Apply normalization
        img_array = img_array / 255.0
        
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
        return img_tensor
        
    except Exception as e:
        st.error(f"Error in image processing: {str(e)}")
        return None

# Improved model loading function
@st.cache_resource
def load_model():
    try:
        # Use the same model structure as original code
        model = keras.models.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
                           input_shape=(224, 224, 3), 
                           trainable=False),
            keras.layers.Dropout(0.2),  # Add dropout to prevent overfitting
            keras.layers.Dense(2, activation='softmax')
        ])
        
        # Compile model with same parameters
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load weights
        try:
            model.load_weights("dog_cat_classifier_weights.h5")
            st.sidebar.success("✅ Model loaded successfully")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Using base model only. Couldn't load weights: {str(e)}")
            
        return model
        
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', ['Prediction', 'About'])

# Add model information in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.info("""
- Base: MobileNetV2
- Training: Transfer Learning
- Classes: Cat (0), Dog (1)
""")

if options == 'Prediction':
    st.title('🐱 Dog vs Cat Classification 🐶')
    st.write("Upload an image of a dog or cat and let the AI identify it!")
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Add confidence threshold slider
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Minimum confidence required for a valid prediction"
        )
        
        # Image upload
        image = st.file_uploader('Upload an image:', type=['jpg', 'jpeg', 'png'])
        
        if image is not None:
            # Display the image
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Create button with better styling
            predict_button = st.button('🔍 Classify Image', key='predict_button', use_container_width=True)
            
            if predict_button:
                if model is not None:
                    with st.spinner('Analyzing...'):
                        try:
                            # Save image bytes for reprocessing if needed
                            image_bytes = image.getvalue()
                            
                            # Process image with enhanced function
                            img_tensor = process_image(io.BytesIO(image_bytes))
                            
                            if img_tensor is not None:
                                # Get prediction
                                predictions = model.predict(img_tensor)[0]
                                predicted_class = predictions.argmax()
                                confidence = predictions[predicted_class]
                                class_names = ['Cat', 'Dog']
                                
                                # Display results in second column
                                with col2:
                                    st.subheader("Classification Results")
                                    
                                    # Create confidence bar
                                    st.write("Confidence Levels:")
                                    for i, class_name in enumerate(class_names):
                                        st.write(f"{class_name}: {predictions[i]:.2%}")
                                        st.progress(float(predictions[i]))
                                    
                                    # Make prediction based on confidence threshold
                                    if confidence >= confidence_threshold:
                                        if predicted_class == 0:
                                            st.markdown(f'<div class="prediction cat">Prediction: Cat</div>', unsafe_allow_html=True)
                                        else:
                                            st.markdown(f'<div class="prediction dog">Prediction: Dog</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="prediction uncertain">Uncertain: Confidence too low</div>', unsafe_allow_html=True)
                                        st.info('Try uploading a clearer image of a cat or dog.')
                                    
                                    # Additional insights
                                    st.markdown("### Image Analysis")
                                    st.write(f"- Most likely class: {class_names[predicted_class]}")
                                    st.write(f"- Confidence: {confidence:.2%}")
                                    
                                    # Provide tips for low confidence results
                                    if confidence < 0.85:
                                        st.markdown("### Tips to improve accuracy")
                                        st.write("- Ensure the animal is clearly visible and centered")
                                        st.write("- Use images with good lighting")
                                        st.write("- Avoid images with multiple animals")
                                        st.write("- Try a different angle or clearer picture")
                                
                        except Exception as e:
                            st.error('Error processing the image. Please make sure you uploaded a valid image file.')
                            st.error(f'Error details: {str(e)}')
                else:
                    st.error("Model could not be loaded. Please check the console for errors.")
    
elif options == 'About':
    st.title('About')
    st.write("""
    This web app is an improved image classification application that uses a pre-trained model to classify images of dogs and cats with enhanced accuracy.
    
    The model uses the MobileNet V2 architecture with transfer learning to achieve high classification accuracy while maintaining fast performance.
    """)
    
    st.subheader("Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 🔍 **Advanced image preprocessing**
        - 🎯 **High classification accuracy**
        - 🛠️ **Transfer learning with MobileNetV2**
        - 📊 **Confidence visualization**
        """)
    
    with col2:
        st.markdown("""
        - ⚙️ **Adjustable confidence threshold**
        - 🖼️ **Support for JPG, JPEG, PNG formats**
        - 🚫 **Uncertain prediction detection**
        - 💻 **Responsive user interface**
        """)
    
    st.subheader("How to Get the Best Results")
    st.write("""
    1. Upload clear, well-lit images of cats or dogs
    2. Ensure the animal is the main subject in the image
    3. Adjust the confidence threshold based on your needs
    4. For ambiguous images, try different angles or lighting
    """)
    
    st.subheader("Technical Details")
    st.write("""
    The model architecture uses MobileNetV2 as a feature extractor with the following improvements:
    
    - Enhanced image preprocessing for better feature extraction
    - Dropout layers to prevent overfitting
    - Center cropping to focus on the main subject
    - High-quality image resizing using LANCZOS algorithm
    """)
    
    st.write('---' * 25)
    st.subheader('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: saadfaisal065@gmail.com')
    st.write('---' * 25)