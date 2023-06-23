# Active Learning demo on Openshift

This is active learning demo using Label Studio and LabelStudio ML backend. This demo trains a model for vegetable classification and also model is actively trained from label studio.

## References

### Requirements

Tested with:
- Python 3.8
- Fedora 38

### Training Data From Kaggle
  
https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset/download?datasetVersionNumber=1

### Model Reference
  
https://www.kaggle.com/code/theeyeschico/vegetable-classification-using-transfer-learning

## Model Training
  
Download Training Data from Kaggle references above and extract the data under model-training/Vegetable Images

### Training the model locally

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-devel.txt
python3 model_training.py
```

### Training the model with RHODS

Use RHODS Project with MinIO Server and establish the data connection and launch Notebook.

Open Notebooks : [notebooks](notebooks)

## Active Learning Label Studio Backend

### Local Active Learning LabelStudioML backend server

```sh
source env/bin/activate
label-studio-ml start serving
```
