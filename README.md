# Active Learning demo on Openshift

This is active learning demo using Label Studio and LabelStudio ML backend. This demo trains a model for vegetable classification and also model is actively trained from label studio.

## About Active Learning

According to Human Signal:

>"To create annotated training data for supervised machine learning models can be expensive and time-consuming. Active Learning is a branch of machine learning that seeks to minimize the total amount of data required for labeling by strategically sampling observations that provide new insight into the problem. In particular, Active Learning algorithms aim to select diverse and informative data for annotation, rather than random observations, from a pool of unlabeled data using prediction scores. For more about the practice of active learning, read this article written by Heartex CTO on Towards Data Science." ~Label Studio Docs

## References

- Use [Label Studio Enterprise Edition](https://docs.humansignal.com/guide/active_learning.html) to build an automated active learning loop with a machine learning model backend.
- If you use the open source [Community Edition of Label Studio](https://docs.humansignal.com/guide/active_learning.html#Customize-your-active-learning-loop), you can manually sort tasks and retrieve predictions to mimic an active learning process.

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
