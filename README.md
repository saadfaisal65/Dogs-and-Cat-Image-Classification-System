# Dogs and Cat Image Classification System 🐶🐱

## Overview

The Dogs and Cat Image Classification System is an interactive web application designed to classify images of cats and dogs. The project leverages Transfer Learning with the MobileNet V2 architecture to deliver accurate predictions efficiently. The interface is built using Streamlit, making it intuitive and user-friendly.

## Features

- **Streamlit-powered Interface**: Easily upload images and get predictions in real-time.
- **MobileNet V2 Model**: A lightweight pre-trained model from TensorFlow Hub for fast and efficient image classification.
- **Custom Fine-tuning**: Added dense layers for binary classification (Dog vs. Cat).
- **Confidence Threshold**: Adjustable threshold for prediction confidence.
- **Error Handling**: Handles invalid or unclear image inputs gracefully.

## Installation

Follow these steps to set up the project:

### 1. Clone the Repository

```bash
git clone https://github.com/saadfaisal65/Dogs-and-Cat-Image-Classification-System.git  
cd Dogs-and-Cat-Image-Classification-System
```

### 2. Set Up a Virtual Environment
#### On Windows
- Create a virtual environment:
  
      python -m venv venv
 
- Activate the virtual environment:

      venv\Scripts\activate

### On macOS/Linux
- Create a virtual environment:

      python3 -m venv venv

- Activate the virtual environment:

      source venv/bin/activate

### 3. Install Dependencies

Once the virtual environment is activated, install the required libraries:

    pip install -r requirements.txt

  The requirements.txt should include the following libraries:

  - **streamlit**
  - **tensorflow**
  - **tensorflow-hub**
  - **numpy**


## Usage
### 1. Run the Application

- To start the Streamlit application, run the following command in the terminal:

      streamlit run app.py

### 2. Use the Web App

  - After running the above command, Streamlit will provide a local URL (e.g., http://localhost:8501/).
  - Open the URL in your browser to access the app.

## How the Project Works
### Model Details

  - Base Model: MobileNet V2 (Pre-trained on ImageNet, frozen for feature extraction).
  - Fine-tuned Layer: A dense output layer with 2 neurons and softmax activation for binary classification.

### Image Processing

Uploaded images are:

- Resized to 224x224 pixels.
- Normalized to a range of [0, 1].
- Expanded to include the batch dimension before feeding into the model.

## App Structure
### Sidebar Navigation

  - Prediction Page:
        - Upload an image for classification.
        - Adjust the confidence threshold for predictions.
  - About Page: Provides details about the model and the app features.

## Key Features

  - Predicts whether the uploaded image is of a dog or a cat.
  - Displays confidence scores for predictions.
  - Alerts if the image is unclear or confidence is below the threshold.

## Contributing

Contributions are welcome! If you would like to improve this project, follow these steps:

  1. Fork the repository.
  2. Create a branch for your feature or bug fix.
  3. Push changes and submit a pull request.

## Contact

For any queries, feedback, or collaboration opportunities:

    Email: saadfaisal065@gmail.com



This version includes all the necessary steps to set up and use the project, including cloning the repo, setting up the virtual environment, installing dependencies, and running the application. Let me know if you'd like any further tweaks!
