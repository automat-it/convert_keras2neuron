import os
import shutil
import zipfile
import boto3
import tensorflow as tf
import numpy as np
from PIL import Image
#import tensorflow.neuron as tfn
import tensorflow_neuronx as tfnx
import subprocess

def download_from_s3(bucket_name, model_key, model_keras_path="model.keras"):
    """
    Download a model from S3.
    
    Args:
        bucket_name (str): AWS S3 bucket name
        model_key (str): S3 key for the model file
        model_keras_path (str, optional): Local path to save the downloaded model. Defaults to "model.keras".
    
    Returns:
        str: Path to the downloaded model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_keras_path) if os.path.dirname(model_keras_path) else '.', exist_ok=True)
    
    # Download model from S3
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, model_key, model_keras_path)
    print(f"Downloaded model from s3://{bucket_name}/{model_key} to {model_keras_path}")
    
    return model_keras_path



def download_zip_s3(bucket_name, model_key, output_dir):
    """
    Download a zip file from S3 and extract its contents.
    
    Args:
        bucket_name (str): AWS S3 bucket name
        model_key (str): S3 key for the zip file
        output_dir (str): Directory to extract the zip contents to
    
    Returns:
        tuple: (output_dir, input_shape)
    """
    # Create temporary file for the zip
    temp_zip_path = os.path.join(os.path.dirname(output_dir) if os.path.dirname(output_dir) else '.', 'temp_download.zip')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download zip from S3 and get object metadata
    s3_client = boto3.client('s3')
    
    # Get object metadata first
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=model_key)
        metadata = response.get('Metadata', {})
        print(f"Retrieved metadata from S3 object: {metadata}")
    except Exception as e:
        print(f"Failed to retrieve metadata: {str(e)}")
        metadata = {}
    
    # Download the zip file
    s3_client.download_file(bucket_name, model_key, temp_zip_path)
    print(f"Downloaded zip from s3://{bucket_name}/{model_key} to {temp_zip_path}")
    
    # Extract the zip file
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted zip contents to {output_dir}")
    
    # Clean up the temporary zip file
    os.remove(temp_zip_path)
    print(f"Removed temporary zip file {temp_zip_path}")
    
    # Try to get input shape from metadata
    input_shape = None
    if 'input-shape' in metadata:
        try:
            # Parse the input shape string from metadata
            input_shape_str = metadata['input-shape']
            # Convert string representation to tuple
            # Example: "(1, 384, 384, 3)" -> (1, 384, 384, 3)
            input_shape_str = input_shape_str.strip('()')
            input_shape = tuple(int(x.strip()) for x in input_shape_str.split(','))
            print(f"Retrieved input shape from metadata: {input_shape}")
        except Exception as e:
            print(f"Failed to parse input shape from metadata: {str(e)}")
    
    # If not found in metadata, try to read from metadata.txt
    if input_shape is None:
        metadata_file = os.path.join(output_dir, "metadata.txt")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    for line in f:
                        if line.startswith("Input Shape:"):
                            input_shape_str = line.replace("Input Shape:", "").strip()
                            # Convert string representation to tuple
                            input_shape_str = input_shape_str.strip('()')
                            input_shape = tuple(int(x.strip()) for x in input_shape_str.split(','))
                            print(f"Retrieved input shape from metadata.txt: {input_shape}")
                            break
            except Exception as e:
                print(f"Failed to parse input shape from metadata.txt: {str(e)}")
    
    return output_dir, input_shape

def convert_model(model_keras_path, saved_model_dir="saved_model", compiled_model_dir="compiled_model"):
    """
    Convert a Keras model to NeuronX format.
    
    Args:
        model_keras_path (str): Path to the Keras model file
        saved_model_dir (str, optional): Directory to save the TensorFlow SavedModel. Defaults to "saved_model".
        compiled_model_dir (str, optional): Directory to save the compiled model. Defaults to "compiled_model".
    
    Returns:
        tuple: (compiled_model_dir, input_shape)
    """
    # Create directories if they don't exist
    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(compiled_model_dir, exist_ok=True)
    
    # Load the Keras model
    model = tf.keras.models.load_model(model_keras_path, compile=False)
    print(f"Loaded Keras model from {model_keras_path}")
    
    # Extract input shape from the model
    input_shape = model.input_shape
    
    if isinstance(input_shape, list):
        input_shape[0] = 1  # Take the first input shape if multiple inputs
    
    if isinstance(input_shape, tuple):
        input_shape = (1, ) + input_shape[1:]  # Take the first input shape if multiple inputs
    
    print(f"Detected input shape: {input_shape}")
    
    # Save as TensorFlow SavedModel
    model.save(saved_model_dir, save_format='tf')
    print(f"Saved model to {saved_model_dir}")
    
    # Compile the model for NeuronX
    
    # Create example input for tracing
    example_input = tf.random.normal(input_shape, dtype=tf.float32)
    
    try:
        # Enable verbose logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        os.environ['NEURON_RT_LOG_LEVEL'] = 'INFO'
        
        # Set verbose if available
        if hasattr(tfnx, 'set_verbose'):
            tfnx.set_verbose(2)
            
        # Trace and compile the model
        neuron_model = tfnx.trace(model, example_input)
        neuron_model.save(compiled_model_dir)
        print("Model compiled successfully")
        print(f"Compiling model for NeuronX to {compiled_model_dir}")
        
        # Test inference with the compiled model
        print("Testing inference with compiled model...")
        output = neuron_model.predict(example_input)
        print(f"Inference successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"Compilation failed: {str(e)}")
        # Fallback: copy the SavedModel to the compiled directory
        print("Using fallback: copying SavedModel to compiled directory")
        for item in os.listdir(saved_model_dir):
            s = os.path.join(saved_model_dir, item)
            d = os.path.join(compiled_model_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    
    # Create a metadata file with the input shape information
    with open(os.path.join(compiled_model_dir, "metadata.txt"), "w") as f:
        f.write(f"Input Shape: {input_shape}\n")
    
    return compiled_model_dir, input_shape

def upload_to_s3(bucket_name, compiled_model_dir, output_zip="neuronx_model.zip", input_shape=None):
    """
    Zip the compiled model and upload it to S3 with metadata.
    
    Args:
        bucket_name (str): AWS S3 bucket name
        compiled_model_dir (str): Directory containing the compiled model
        input_shape (tuple, optional): Input shape of the model. If None, will try to read from metadata.txt
        output_zip (str, optional): Output zip file name. Defaults to "neuronx_model.zip".
    
    Returns:
        str: S3 key of the uploaded zip file
    """
    # Zip the compiled model
    shutil.make_archive(output_zip.replace('.zip', ''), 'zip', compiled_model_dir)
    print(f"Created zip file {output_zip}")
    
    # Prepare metadata
    metadata = {}
    
    # If input_shape is provided, use it
    if input_shape is not None:
        # Convert input shape to string representation for metadata
        input_shape_str = str(input_shape)
        metadata['Input-Shape'] = input_shape_str
        print(f"Added input shape metadata: {input_shape_str}")
    else:
        # Try to read from metadata.txt if exists
        metadata_file = os.path.join(compiled_model_dir, "metadata.txt")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                for line in f:
                    if line.startswith("Input Shape:"):
                        input_shape_str = line.replace("Input Shape:", "").strip()
                        metadata['Input-Shape'] = input_shape_str
                        print(f"Added input shape metadata from file: {input_shape_str}")
                        break
    
    # Add timestamp metadata
    from datetime import datetime
    metadata['Creation-Date'] = datetime.now().isoformat()
    
    # Upload to S3 with metadata
    s3_client = boto3.client('s3')
    s3_client.upload_file(
        output_zip, 
        bucket_name, 
        output_zip,
        ExtraArgs={'Metadata': metadata}
    )
    print(f"Uploaded {output_zip} to S3 bucket {bucket_name} with metadata")
    print(f"Uploaded to s3://{bucket_name}/{output_zip}")
    
    return output_zip

def get_segmentation(image_path, model_dir, input_shape=None):
    """
    Generate a segmentation mask for an image using a model.
    
    Args:
        image_path (str): Path to the input image
        model_dir (str): Directory containing the model
    
    Returns:
        numpy.ndarray: Segmentation mask
    """
    try:
        # Load the model
        model = tf.keras.models.load_model(model_dir)
        print(f"Loaded model from {model_dir}")
        
        # Determine input shape if not provided
        if input_shape is None:
            # Try to get from model
            if hasattr(model, 'input_shape'):
                input_shape = model.input_shape
                if isinstance(input_shape, list):
                    input_shape[0] = 1
                if isinstance(input_shape, tuple):
                    input_shape = (1, ) + input_shape[1:]
            else:
                # Default to a common image size
                input_shape = (1, 384, 384, 3)
            print("Using input shape", input_shape)
        
        # Load and preprocess the image
        input_image = Image.open(image_path).convert("RGB")
        
        input_size = (input_shape[1], input_shape[2])
        input_image = input_image.resize(input_size, Image.BILINEAR)
        print(f"Loaded and preprocessed image from {image_path}")
        
        # Normalize the image (using common ImageNet normalization)
        input_array = np.array(input_image, dtype=np.float32) / 255.0
        input_array = (input_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        input_tensor = np.expand_dims(input_array, axis=0)
        
        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
        
        # Run inference
        print("Running inference for segmentation...")
        output = model.predict(input_tensor)
        print(f"Inference successful, output shape: {output.shape}")
        
        # Process the output to get segmentation mask
        if len(output.shape) == 4 and output.shape[3] > 1:
            # Multi-class segmentation (apply argmax to get class with highest probability)
            segmentation_mask = np.argmax(output[0], axis=-1)
            print(f"Generated multi-class segmentation mask with shape: {segmentation_mask.shape}")
            print(f"Unique class labels: {np.unique(segmentation_mask)}")
        else:
            # Binary segmentation (threshold at 0.5)
            segmentation_mask = (output[0, ..., 0] > 0.5).astype(np.uint8)
            print(f"Generated binary segmentation mask with shape: {segmentation_mask.shape}")
        
        return segmentation_mask
        
    except Exception as e:
        print(f"Segmentation failed: {str(e)}")
        return None

def check_inference(model_dir, image_path, input_shape=None):
    """
    Check inference with the compiled model using a test image.
    
    Args:
        compiled_model_dir (str): Directory containing the compiled model
        image_path (str): Path to the test image
        input_shape (tuple, optional): Input shape for the model. If None, will be determined from the model.
    
    Returns:
        tuple: (success, output)
    """
    try:
        # Load the compiled model
        neuron_model = tf.keras.models.load_model(model_dir)
        print(f"Loaded compiled model from {model_dir}")
        
        # Determine input shape if not provided
        if input_shape is None:
            # Try to get from model
            if hasattr(neuron_model, 'input_shape'):
                input_shape = neuron_model.input_shape
                if isinstance(input_shape, list):
                    input_shape[0] = 1
                if isinstance(input_shape, tuple):
                    input_shape = (1, ) + input_shape[1:]
            else:
                # Default to a common image size
                input_shape = (1, 384, 384, 3)
        
        # Load and preprocess the image
        input_image = Image.open(image_path).convert("RGB")
        input_size = (input_shape[1], input_shape[2])
        input_image = input_image.resize(input_size, Image.BILINEAR)
        print(f"Loaded and preprocessed image from {image_path}")
        
        # Normalize the image (using common ImageNet normalization)
        input_array = np.array(input_image, dtype=np.float32) / 255.0
        input_array = (input_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        input_tensor = np.expand_dims(input_array, axis=0)
        
        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
        
        # Run inference
        print("Running inference...")
        output = neuron_model.predict(input_tensor)
        print(f"Inference successful, output shape: {output.shape}")
        
        # If output is a segmentation mask (has channel dimension with classes)
        if len(output.shape) == 4 and output.shape[3] > 1:
            # Apply argmax to get the class with highest probability
            segmentation_mask = np.argmax(output[0], axis=-1)
            print(f"Segmentation mask shape: {segmentation_mask.shape}")
            print(f"Unique class labels: {np.unique(segmentation_mask)}")
        
        return True, output
        
    except Exception as e:
        print(f"Inference check failed: {str(e)}")
        return False, None

def convert_to_neuron(
    bucket_name,
    model_key,
    image_path,
    model_keras_path="model.keras",
    saved_model_dir="saved_model",
    compiled_model_dir="compiled_model",
    output_zip="neuronx_model.zip"
):
    """
    Main function to convert a Keras model to Neuron format and test it.
    
    Args:
        bucket_name (str): AWS S3 bucket name
        model_key (str): S3 key for the model file
        image_path (str): Path to a test image for inference
        model_keras_path (str, optional): Local path to save the downloaded model. Defaults to "model.keras".
        saved_model_dir (str, optional): Directory to save the TensorFlow SavedModel. Defaults to "saved_model".
        compiled_model_dir (str, optional): Directory to save the compiled model. Defaults to "compiled_model".
        output_zip (str, optional): Output zip file name. Defaults to "neuronx_model.zip".
    
    Returns:
        dict: Results of the conversion process
    """
    results = {
        "success": False,
        "model_path": None,
        "s3_path": None,
        "inference_success": False
    }
    
    try:
        # Check Neuron device status
        subprocess.run(["neuron-ls"])
        print("Checked Neuron device status")
        
        # Step 1: Download from S3
        model_path = download_from_s3(bucket_name, model_key, model_keras_path)
        
        # Step 2: Convert model
        compiled_dir, input_shape = convert_model(model_path, saved_model_dir, compiled_model_dir)
        results["model_path"] = compiled_dir
        print(f"Converted model to {compiled_dir} and {input_shape}")

        # Step 3: Check inference
        inference_success, _ = check_inference(compiled_dir, image_path, input_shape)
        results["inference_success"] = inference_success
        
        # Step 4: Upload to S3
        s3_path = upload_to_s3(bucket_name, compiled_dir, output_zip, input_shape)
        results["s3_path"] = f"s3://{bucket_name}/{s3_path}"
        
        results["success"] = True
        print("Conversion process completed successfully")
        
    except Exception as e:
        print(f"Conversion process failed: {str(e)}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Keras model to NeuronX format')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--model-key', required=True, help='S3 key for the model file')
    parser.add_argument('--image-path', required=True, help='Path to a test image for inference')
    parser.add_argument('--keras-path', default='model.keras', help='Local path to save the downloaded Keras model')
    parser.add_argument('--saved-model-dir', default='saved_model', help='Directory to save the TensorFlow SavedModel')
    parser.add_argument('--compiled-model-dir', default='compiled_model', help='Directory to save the compiled model')
    parser.add_argument('--output-zip', default='neuronx_model.zip', help='Output zip file name')
    
    args = parser.parse_args()
    
    results = convert_to_neuronx(
        args.bucket,
        args.model_key,
        args.image_path,
        args.keras_path,
        args.saved_model_dir,
        args.compiled_model_dir,
        args.output_zip
    )
    
    print("\nConversion Results:")
    for key, value in results.items():
        print(f"{key}: {value}")