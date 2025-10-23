🐟 Multiclass Fish Image Classification
🔹 Project Overview

A deep learning project to classify fish species from images.
The workflow includes:

Training a CNN from scratch

Fine-tuning pre-trained models: VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0

Deploying a Streamlit app for real-time predictions
<img width="477" height="762" alt="image" src="https://github.com/user-attachments/assets/767a60c3-732d-4ce0-b45a-66cf04a72ead" />



🛠 Technologies & Skills
Category	Tools / Libraries
Programming	Python 3.11
Deep Learning	TensorFlow, Keras
Web Deployment	Streamlit
Data Processing	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Concepts	CNN, Transfer Learning, Data Augmentation
🎯 Problem Statement

Build a robust system to classify fish images into multiple species with high accuracy.

Business Use Cases:

Accurate Species Classification for fisheries and research

User-Friendly Web Application for real-time predictions

Model Comparison to select the best architecture

📂 Dataset

Format: .jpg images

Structure: Images grouped by species in separate folders

Loading: TensorFlow ImageDataGenerator for preprocessing and augmentation

🧰 Approach
1️⃣ Data Preprocessing & Augmentation

Rescale images to [0,1]

Apply augmentation: rotation, zoom, flips

2️⃣ Model Training

Train CNN from scratch

Fine-tune pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0)

Save the best model in .h5 or .pkl

3️⃣ Model Evaluation

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Visualizations: Training & Validation accuracy/loss

Example Training Plot:

<!-- Replace with your plot -->

4️⃣ Deployment with Streamlit

Upload a fish image and predict species

Display prediction and confidence scores

App Screenshot:

<!-- Replace with your screenshot -->

📈 Model Comparison
Model	Accuracy	Precision	Recall	F1-Score
CNN (Scratch)	85%	84%	85%	84%
VGG16	90%	89%	90%	89%
ResNet50	92%	91%	92%	91%
MobileNet	89%	88%	89%	88%
InceptionV3	91%	90%	91%	90%
EfficientNetB0	93%	92%	93%	92%
🚀 How to Run
1️⃣ Clone the Repository
git clone <repository_url>
cd <repository_folder>

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py

4️⃣ Upload Fish Image

Drag and drop or select an image

Get predicted species and confidence score
