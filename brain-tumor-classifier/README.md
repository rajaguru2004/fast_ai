# Brain Tumor MRI Classifier

This repository contains a simple, interactive web application to classify brain MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

The model was trained using the `fastai` library on a ResNet-18 architecture.

## Folder Structure

```
brain-tumor-classifier/
│
├── app.py                     # Gradio web application
├── requirements.txt           # Python package dependencies
├── brain_tumor_classifier.pkl # Exported model weights
├── README.md                  # Project documentation
├── .gitignore                 # Files to ignore in git
└── assets/                    # Directory containing sample images
```

## Setup and Running the Web App

1. **Activate your environment** (if not already activated):
   ```bash
   source ../.venv/bin/activate
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Gradio app**:
   ```bash
   python app.py
   ```

After running the command, open the local URL (usually `http://127.0.0.1:7860`) in your browser to interact with the application.
