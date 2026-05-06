# ASL Image Classification with CLIP Fine-Tuning

## Overview

This project builds an American Sign Language (ASL) image classification system using a fine-tuned CLIP vision model. The goal is to recognize ASL hand-sign images for:

- digits `0` to `9`
- letters `a` to `z`

The project includes:

- a training notebook developed in Google Colab
- a saved model checkpoint
- a local prediction script
- a local desktop UI for image-based inference

## Project Files

- [Training-Model.ipynb](C:/Users/prish/PycharmProjects/VAI/CLIP/Training-Model.ipynb)  
  Google Colab notebook used to train and evaluate the model.

- [asl_predict.py](C:/Users/prish/PycharmProjects/VAI/CLIP/asl_predict.py)  
  Python inference script with a Tkinter UI and text-to-speech support.

- [clip_asl_ft.pth](C:/Users/prish/PycharmProjects/VAI/CLIP/clip_asl_ft.pth)  
  Best saved fine-tuned model checkpoint used for inference.

- [asl_dataset](C:/Users/prish/PycharmProjects/VAI/CLIP/asl_dataset)  
  Dataset folder used for training and evaluation.

- [training-ouput](C:/Users/prish/PycharmProjects/VAI/CLIP/training-ouput)  
  Training result plots such as accuracy, loss, and weighted F1 score.

## Training

Model training was performed in the Colab notebook [Training-Model.ipynb](C:/Users/prish/PycharmProjects/VAI/CLIP/Training-Model.ipynb).

In the notebook, the workflow is:

1. load the ASL dataset
2. apply image preprocessing and transformations
3. initialize the CLIP-based classifier
4. fine-tune the model on the ASL classes
5. evaluate performance during training
6. save model checkpoints, including the best model

The notebook is the main training component of the project.

## Model Architecture

The project uses:

- `openai/clip-vit-base-patch32`
- a custom classification head:
  - `Dropout(0.2)`
  - `Linear(projection_dim, num_classes)`

The classifier predicts one class from the combined set of:

- 26 alphabet classes

Total classes: `36`

## Inference Flow

The inference application loads the saved checkpoint, preprocesses an input image, runs the model, and returns:

- predicted class
- confidence score

### Model Used for Prediction

The current prediction script uses:

```python
MODEL_PATH = "clip_asl_ft.pth"
```

This is the model currently used for inference in:

- [asl_predict.py](C:/Users/prish/PycharmProjects/VAI/CLIP/asl_predict.py)

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Make sure a model file is present

The default model file used by the project is:

- `clip_asl_ft.pth`

### 3. Run the prediction UI

```bash
python asl_predict.py
```

### 4. Select an image

Click **Choose Image**, select an ASL image, and the application will display:

- predicted sign
- confidence score

In `asl_predict.py`, the prediction is also spoken aloud using `pyttsx3`.

## Dependencies

The project uses the following Python packages:

- `torch`
- `torchvision`
- `transformers`
- `pillow`
- `pyttsx3`

## Output

The training results are stored in [training-ouput](C:/Users/prish/PycharmProjects/VAI/CLIP/training-ouput), including:

- `Accuracy.png`
- `Loss.png`
- `WeightedF1.png`

These plots help summarize training performance.

## Educational Note

This project is intended for educational and demonstration purposes. It shows how a pretrained vision-language model such as CLIP can be fine-tuned for a domain-specific classification task and then used in a local prediction application.

## Repository Note

The trained model checkpoint and local dataset are not included in the GitHub repository. The file clip_asl_ft.pth is about 577 MB, which exceeds standard GitHub file limits, and the dataset is kept locally for training and testing.

