# LungDetect-AI

A Deep Learning web application built using **Flask** and **PyTorch** that predicts chest X-ray diseases from X-Ray images.

---

## About the Project

This is a deep learning project that demonstrates:

- Image preprocessing
- CNN model architecture implementation
- Model training in Jupyter Notebook
- Saving trained model using PyTorch (.pth)
- Deploying model using Flask API
- Building a simple authentication-based web interface

The model predicts:

- **Covid-19**
- **Emphysema**
- **Normal**
- **Pneumonia**
- **Tuberculosis**

---

## Technologies Used

- Python
- Flask
- PyTorch
- Torchvision
- NumPy
- Flask-CORS
- Jupyter Notebook
- Jason
- HTML
- CSS

---

## Project Structure

    ├── app.py
    ├── server.py
    ├── Net.py
    ├── best_model1.pth
    ├── notebook3.ipynb
    ├── users.json
    ├── README.md

---

## Installation & Setup

### 1. Clone the repository

    git clone https://github.com/Ravin141/LungDetect-AI.git
    cd LungDetect-AI

### 2. Install dependencies

    pip install flask torch torchvision numpy pillow flask-cors

### 3. Run the application

Start the prediction API:

    python server.py

Start the frontend application:

    python app.py

---

## Features

User can:

- Register and Login
- Upload Chest X-ray image
- Get predicted disease label
- Receive prediction confidence score
- Access dashboard interface

Click **Upload & Predict** to see the result.

---

## Future Improvements

- Add probability visualization chart
- Show model evaluation metrics
- Improve UI design
- Deploy online
- Add database integration
- Add Grad-CAM heatmap visualization

---

## Author

Ravin Perera  
Undergraduate – BSc (Hons) Software Engineering
