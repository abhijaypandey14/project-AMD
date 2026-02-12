# project-AMD
Project AMD is a deep learning initiative designed to assist in the early detection and classification of Age-Related Macular Degeneration (AMD). Using Convolutional Neural Networks (CNNs) and retinal fundus images, this system automates the diagnosis process to help medical professionals identify the disease stages (e.g., Normal, Dry AMD, Wet AMD).

🎯 Objectives
To preprocess and augment medical retinal datasets.

To build and train a CNN model (e.g., ResNet, VGG19, or Custom CNN) for image classification.

To achieve high sensitivity and specificity in detecting macular degeneration.

To provide a user-friendly interface for uploading and testing images.

📂 Dataset
This project utilizes retinal fundus images.

Source: [Insert Source, e.g., ODIR-5K, Kaggle AMD Dataset, or RFMiD]

Classes:

Normal

AMD (Early/Late/Wet/Dry)

Note: Due to privacy restrictions, the dataset is not included in this repository. Please download it from [Link] and place it in the data/ folder.

🛠️ Tech Stack
Language: Python

Libraries:

TensorFlow / Keras or PyTorch (Deep Learning)

OpenCV (Image Processing)

NumPy & Pandas (Data Manipulation)

Matplotlib & Seaborn (Visualization)

Streamlit or Flask (Web Interface - if applicable)

🚀 Getting Started
Follow these steps to set up the project locally.

Prerequisites
Python 3.8 or higher

Git

Installation
Clone the repository

Bash
git clone https://github.com/abhijaypandey14/project-AMD.git
cd project-AMD
Create a Virtual Environment (Optional but recommended)

Bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
Install Dependencies

Bash
pip install -r requirements.txt
🧠 Usage
1. Training the Model
To retrain the model with your own dataset:

Bash
python train.py --epochs 20 --batch_size 32
2. Testing/Inference
To test the model on a single image:

Bash
python predict.py --image "path/to/image.jpg"
3. Running the Web App (If applicable)
If the project includes a UI:

Bash
streamlit run app.py
📊 Results
Accuracy: 9X.XX%

Loss: 0.XX

Confusion Matrix: (You can upload a screenshot of your confusion matrix or training graphs here)

🔮 Future Scope
Integration with mobile applications for remote diagnosis.

Implementing Explainable AI (Grad-CAM) to visualize which parts of the retina the model is looking at.

Expanding the dataset to include other retinal diseases (Glaucoma, Diabetic Retinopathy).

🤝 Contributing
Contributions are welcome!

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

👤 Author
Abhijay Pandey

GitHub: @abhijaypandey14

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
