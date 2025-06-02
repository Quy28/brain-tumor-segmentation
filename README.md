dataset để train **BraTS2020 Dataset** (Training + Validation): awsaf49/brats20-dataset-training-validation
🧠 Brain Tumor Segmentation with Attention U-Net (BraTS 2020)
This project implements a brain tumor segmentation pipeline using the BraTS 2020 dataset and an Attention U-Net model. The system is modularized into 4 key Python files for training, predicting, and visualizing tumor regions.

🧰 Requirements
Install dependencies:
pip install tensorflow nibabel opencv-python matplotlib
🧪 Training the Model
from train import train_model, plot_training
from your_model_module import get_model  # Define your model here
model = get_model()  # Your Attention U-Net model
history = train_model(model, training_generator, validation_generator, train_ids)
plot_training(history)
🔮 Predict on New Case
from predict import predictByPath
from keras.models import load_model
model = load_model("unet_model.h5", compile=False)
prediction = predictByPath(case_path, case_id, model)
🖼️ Visualize Predictions
from visualize import showPredictsById
showPredictsById(case='123', model=model, predict_fn=predictByPath)
✅ This shows 6 images: original slice, ground truth, all predicted classes, and individual tumor types (Edema, Core, Enhancing Tumor).
🧠 Label Classes
Label	Class Name
0	Background
1	Edema
2	Non-Enhancing Core
3	Enhancing Tumor
