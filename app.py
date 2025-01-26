import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras
import numpy as np

# Model loading
loaded_model = keras.models.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", input_shape=(224,224,3), trainable=False),
    keras.layers.Dense(2, activation='softmax')  # Added softmax activation for probability output
])
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
loaded_model.load_weights("dog_cat_classifier_weights.h5")

def process_image(image):
    img = keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)  
    return img

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                           ['Prediction', 'About'])

if options == 'Prediction':
    st.title('Dog vs Cat Classification with Transfer Learning')
    
    # Add confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Minimum confidence required for a valid prediction"
    )
    
    # User inputs: image
    image = st.file_uploader('Upload an image:', type=['jpg', 'jpeg', 'png'])
    if image is not None:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        with st.spinner('Model working....'):
            try:
                img_array = process_image(image)
                predictions = loaded_model.predict(img_array)[0]
                predicted_class = predictions.argmax()
                confidence = predictions[predicted_class]
                
                              
                # Make prediction based on confidence threshold
                if confidence >= confidence_threshold:
                    if predicted_class == 0:
                        st.success('Prediction: Cat')
                    else:
                        st.success('Prediction: Dog')
                else:
                    st.warning('This image does not appear to be a cat or a dog with sufficient confidence.')
                    st.info('Try uploading a clearer image of a cat or dog.')
                
            except Exception as e:
                st.error('Error processing the image. Please make sure you uploaded a valid image file.')
                st.error(f'Error details: {str(e)}')
            
elif options == 'About':
    st.title('About')
    st.write('This web app is a simple image classification app that uses a pre-trained model to classify images of dogs and cats.')
    st.write('The model is trained using the MobileNet V2 architecture with ImageNet pre-trained weights.')
    st.write('Features:')
    st.write('• Classifies images as either cats or dogs')
    st.write('• Provides confidence scores for predictions')
    st.write('• Detects when an image is neither a cat nor a dog')
    st.write('• Adjustable confidence threshold for predictions')
    st.write('--'*50)
    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('--'*50)
    st.header('Contact')
    st.write('You can contact me for any queries or feedback:')
    st.write('Email: saadfaisal065@gmail.com')
    st.write('--'*50)