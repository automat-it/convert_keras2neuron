# Convert2NeuronX

A utility for converting TensorFlow/Keras models to AWS Neuron format for deployment on AWS Inferentia and Trainium instances.

## Overview

This project provides tools to convert deep learning models (particularly segmentation models) from standard TensorFlow/Keras format to AWS Neuron format, which is optimized for AWS Inferentia and Trainium accelerators. The conversion process includes:

1. Downloading models from S3
2. Converting to NeuronX format
3. Testing inference
4. Packaging and uploading the compiled model back to S3

## Features

- Automatic model conversion from Keras to NeuronX format
- Preservation of input shape metadata
- Inference testing with sample images
- S3 integration for model storage and retrieval
- Segmentation mask generation from compiled models

## Requirements

- Python 3.x
- TensorFlow 2.x
- AWS Neuron SDK
- AWS SDK for Python (boto3)
- PIL (Python Imaging Library)
- An AWS account with access to S3
- AWS Inferentia or Trainium instance inf2.x

## Installation

Clone this repository and install the required dependencies:

```bash

pip install tensorflow tensorflow-neuronx boto3 pillow
```

## Usage
### Basic Conversion
```python
from convert2neuronx import convert_to_neuron

# Convert a model
result = convert_to_neuron(
    bucket_name='your-s3-bucket',
    model_key='path/to/model.keras',
    image_path='path/to/test_image.jpg'
)

print(result)
```
### Loading a Compiled Model

```python
from convert2neuronx import download_zip_s3, get_segmentation

# Download and load a compiled model
model_dir, input_shape = download_zip_s3(
    bucket_name='your-s3-bucket',
    model_key='path/to/compiled_model.zip',
    output_dir='./model_dir'
)

# Generate a segmentation mask
mask = get_segmentation('path/to/image.jpg', model_dir, input_shape)
```
## Main Functions
- download_from_s3 : Downloads a model from S3
- convert_model : Converts a Keras model to NeuronX format
- upload_to_s3 : Packages and uploads a compiled model to S3
- get_segmentation : Generates a segmentation mask using a compiled model
- download_zip_s3 : Downloads and extracts a compiled model from S3
- convert_to_neuron : Main function that orchestrates the entire conversion process
## Example Notebooks
The repository includes several Jupyter notebooks demonstrating the conversion and inference process:

- convert2neuronx.ipynb : Demonstrates the basic conversion process
- Check_python_module.ipynb : Tests the Python module functionality
- Test_Neuronx.ipynb : Tests inference with compiled models
- Tect_python.ipynb : Additional testing and examples
## License
[Specify your license here]
```plaintext

This README provides a comprehensive overview of your Convert2NeuronX project, explaining its purpose, features, and usage. It includes information about the main functions available in your module and examples of how to use them.

Would you like me to make any adjustments to this README?
```
# convert-keras2neuron
