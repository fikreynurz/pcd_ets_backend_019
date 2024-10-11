# app.py

import os
from uuid import uuid4
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import io
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS Configuration - Allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Update jika frontend berjalan di host atau port lain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure necessary directories exist
required_dirs = [
    "static/histograms",
    "static/uploads",
    "dataset",
    "processed_dataset"
]

for directory in required_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def read_image(uploaded_file: bytes) -> np.ndarray:
    """Read image from uploaded bytes."""
    image_data = np.frombuffer(uploaded_file, np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return img

def encode_image(image: np.ndarray) -> bytes:
    """Encode image to PNG bytes."""
    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        logger.error("Failed to encode image to PNG format.")
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    return encoded_image.tobytes()

def save_histogram(image: np.ndarray, prefix: str) -> str:
    """Save grayscale or color histogram with consistent UUID."""
    unique_id = uuid4().hex  # Generate unique identifier once
    histogram_path = f"static/histograms/{prefix}_{unique_id}.png"
    
    plt.figure()
    if prefix == "grayscale":
        plt.hist(image.ravel(), 256, [0, 256])
    elif prefix == "color":
        for i, color in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
    plt.savefig(histogram_path)
    plt.close()
    
    logger.info(f"Saved {prefix} histogram at: {histogram_path}")
    return f"/static/histograms/{prefix}_{unique_id}.png"

# Endpoint: Convert to Grayscale
@app.post("/grayscale")
async def convert_to_grayscale(file: UploadFile = File(...)):
    """
    Convert uploaded image to grayscale.
    Returns the grayscale image.
    """
    try:
        if file is None:
            logger.warning("No file uploaded for grayscale conversion.")
            raise HTTPException(status_code=400, detail="File cannot be empty.")
        
        if not file.content_type.startswith('image/'):
            logger.warning(f"Uploaded file is not an image: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image.")
        
        contents = await file.read()
        img = read_image(contents)
    
        if img is None:
            logger.error("Failed to read uploaded image for grayscale conversion.")
            raise HTTPException(status_code=400, detail="Cannot read the uploaded image.")
    
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        return StreamingResponse(
            io.BytesIO(encode_image(gray_img)),
            media_type="image/png"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /grayscale endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Endpoint: Generate Histograms
@app.post("/histogram")
async def generate_histogram(file: UploadFile = File(...)):
    """
    Generate grayscale and color histograms from uploaded image.
    Returns the paths to the saved histograms.
    """
    try:
        if file is None:
            logger.warning("No file uploaded for histogram generation.")
            raise HTTPException(status_code=400, detail="File cannot be empty.")
        
        if not file.content_type.startswith('image/'):
            logger.warning(f"Uploaded file is not an image: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image.")
        
        contents = await file.read()
        img = read_image(contents)
    
        if img is None:
            logger.error("Failed to read uploaded image.")
            raise HTTPException(status_code=400, detail="Cannot read the uploaded image.")
    
        # Generate Grayscale Histogram
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale_histogram_path = save_histogram(gray_img, "grayscale")
    
        # Generate Color Histogram
        color_histogram_path = save_histogram(img, "color")
    
        logger.info(f"Grayscale histogram path: {grayscale_histogram_path}")
        logger.info(f"Color histogram path: {color_histogram_path}")
    
        return JSONResponse(content={
            "grayscale_histogram": grayscale_histogram_path,
            "color_histogram": color_histogram_path
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /histogram endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Endpoint: RGB Extraction
@app.post("/rgb_extraction")
async def rgb_extraction(file: UploadFile = File(...)):
    """
    Extract average RGB values from uploaded image.
    Returns the average RGB values.
    """
    try:
        if file is None:
            logger.warning("No file uploaded for RGB extraction.")
            raise HTTPException(status_code=400, detail="File cannot be empty.")
        
        if not file.content_type.startswith('image/'):
            logger.warning(f"Uploaded file is not an image: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image.")
        
        contents = await file.read()
        img = read_image(contents)
    
        if img is None:
            logger.error("Failed to read uploaded image for RGB extraction.")
            raise HTTPException(status_code=400, detail="Cannot read the uploaded image.")
    
        # Calculate average RGB
        average_rgb = img.mean(axis=(0,1)).tolist()  # [B, G, R]
        average_rgb = average_rgb[::-1]  # Convert to [R, G, B]
    
        logger.info(f"Extracted average RGB: {average_rgb}")
    
        return JSONResponse(content={"average_rgb": average_rgb})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /rgb_extraction endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Endpoint: Add Noise
@app.post("/add_noise")
async def add_noise(file: UploadFile = File(...), noise_prob: float = 0.05, salt_color: str = "#FFFFFF", pepper_color: str = "#000000"):
    """
    Add salt and pepper noise to uploaded image.
    Returns the noisy image.
    """
    try:
        if file is None:
            logger.warning("No file uploaded for adding noise.")
            raise HTTPException(status_code=400, detail="File cannot be empty.")
        
        if not file.content_type.startswith('image/'):
            logger.warning(f"Uploaded file is not an image: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image.")
        
        if not (0.0 <= noise_prob <= 1.0):
            logger.warning(f"Invalid noise probability: {noise_prob}")
            raise HTTPException(status_code=400, detail="Noise probability must be between 0.0 and 1.0.")
        
        contents = await file.read()
        img = read_image(contents)
    
        if img is None:
            logger.error("Failed to read uploaded image for adding noise.")
            raise HTTPException(status_code=400, detail="Cannot read the uploaded image.")
    
        # Convert salt and pepper colors from hex to BGR
        salt_bgr = tuple(int(salt_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        pepper_bgr = tuple(int(pepper_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    
        # Add salt and pepper noise
        noisy_img = img.copy()
        num_salt = np.ceil(noise_prob * img.size * 0.5).astype(int)
        num_pepper = np.ceil(noise_prob * img.size * 0.5).astype(int)
    
        # Salt noise
        coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_img.shape[:2]]
        noisy_img[coords[0], coords[1]] = salt_bgr
    
        # Pepper noise
        coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_img.shape[:2]]
        noisy_img[coords[0], coords[1]] = pepper_bgr
    
        # Encode image to bytes
        encoded_image = encode_image(noisy_img)
    
        logger.info(f"Added salt and pepper noise with probability {noise_prob}")
    
        return StreamingResponse(
            io.BytesIO(encoded_image),
            media_type="image/png"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /add_noise endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Endpoint: Remove Noise
@app.post("/remove_noise")
async def remove_noise(file: UploadFile = File(...)):
    """
    Remove salt and pepper noise using median filter.
    Returns the denoised image.
    """
    try:
        if file is None:
            logger.warning("No file uploaded for removing noise.")
            raise HTTPException(status_code=400, detail="File cannot be empty.")
        
        if not file.content_type.startswith('image/'):
            logger.warning(f"Uploaded file is not an image: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image.")
        
        contents = await file.read()
        img = read_image(contents)
    
        if img is None:
            logger.error("Failed to read uploaded image for removing noise.")
            raise HTTPException(status_code=400, detail="Cannot read the uploaded image.")
    
        # Apply median filter
        denoised_img = cv2.medianBlur(img, 3)
    
        # Encode image to bytes
        encoded_image = encode_image(denoised_img)
    
        logger.info("Applied median filter to remove noise.")
    
        return StreamingResponse(
            io.BytesIO(encoded_image),
            media_type="image/png"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /remove_noise endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Endpoint: Sharpen Image
@app.post("/sharpen_image")
async def sharpen_image(file: UploadFile = File(...)):
    """
    Sharpen the uploaded image using a kernel.
    Returns the sharpened image.
    """
    try:
        if file is None:
            logger.warning("No file uploaded for sharpening image.")
            raise HTTPException(status_code=400, detail="File cannot be empty.")
        
        if not file.content_type.startswith('image/'):
            logger.warning(f"Uploaded file is not an image: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image.")
        
        contents = await file.read()
        img = read_image(contents)
    
        if img is None:
            logger.error("Failed to read uploaded image for sharpening.")
            raise HTTPException(status_code=400, detail="Cannot read the uploaded image.")
    
        # Define sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharpened_img = cv2.filter2D(img, -1, kernel)
    
        # Encode image to bytes
        encoded_image = encode_image(sharpened_img)
    
        logger.info("Applied sharpening filter to image.")
    
        return StreamingResponse(
            io.BytesIO(encoded_image),
            media_type="image/png"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /sharpen_image endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Endpoint: RGB Operations
@app.post("/operation_rgb")
async def operation_rgb(file: UploadFile = File(...),
    operation: str = Form(...),
    value: int = Form(...)):
    """
    Perform arithmetic operations on RGB channels.
    Supported operations: add, subtract, max, min, inverse
    Returns the processed image.
    """
    try:
        if file is None:
            logger.warning("No file uploaded for RGB operation.")
            raise HTTPException(status_code=400, detail="File cannot be empty.")
        
        if not file.content_type.startswith('image/'):
            logger.warning(f"Uploaded file is not an image: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image.")
        
        contents = await file.read()
        img = read_image(contents)
    
        if img is None:
            logger.error("Failed to read uploaded image for RGB operation.")
            raise HTTPException(status_code=400, detail="Cannot read the uploaded image.")
    
        # Perform operation
        if operation == 'add':
            processed_img = cv2.add(img, np.array([value]))
        elif operation == 'subtract':
            processed_img = cv2.subtract(img, np.array([value]))
        elif operation == 'max':
            processed_img = cv2.max(img, np.array([value]))
        elif operation == 'min':
            processed_img = cv2.min(img, np.array([value]))
        elif operation == 'inverse':
            processed_img = cv2.bitwise_not(img)
        else:
            logger.warning(f"Unsupported operation: {operation}")
            raise HTTPException(status_code=400, detail="Unsupported operation.")
    
        # Encode image to bytes
        encoded_image = encode_image(processed_img)
    
        logger.info(f"Applied {operation} operation with value {value}.")
    
        return StreamingResponse(
            io.BytesIO(encoded_image),
            media_type="image/png"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /operation_rgb endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Endpoint: Logical Operations on RGB Images
@app.post("/logic_operation_rgb")
async def logic_operation_rgb(file1: UploadFile = File(...), file2: Optional[UploadFile] = File(...), operation: str = Form(...)):
    """
    Perform logical operations on two RGB images.
    Supported operations: and, xor, not
    For 'not', only file1 is used.
    Returns the processed image.
    """
    try:
        if file1 is None:
            logger.warning("No main file uploaded for logical operation.")
            raise HTTPException(status_code=400, detail="Main file cannot be empty.")
        
        if not file1.content_type.startswith('image/'):
            logger.warning(f"Uploaded main file is not an image: {file1.content_type}")
            raise HTTPException(status_code=400, detail="Main file must be an image.")
        
        if operation in ['and', 'xor'] and file2 is None:
            logger.warning(f"No reference file uploaded for {operation} operation.")
            raise HTTPException(status_code=400, detail=f"Reference file is required for {operation} operation.")
        
        if operation in ['and', 'xor'] and file2 and not file2.content_type.startswith('image/'):
            logger.warning(f"Uploaded reference file is not an image: {file2.content_type}")
            raise HTTPException(status_code=400, detail="Reference file must be an image.")
        
        # Read main image
        contents1 = await file1.read()
        img1 = read_image(contents1)
        if img1 is None:
            logger.error("Failed to read main uploaded image for logical operation.")
            raise HTTPException(status_code=400, detail="Cannot read the main uploaded image.")
        
        # Read reference image if needed
        if operation in ['and', 'xor']:
            contents2 = await file2.read()
            img2 = read_image(contents2)
            if img2 is None:
                logger.error("Failed to read reference uploaded image for logical operation.")
                raise HTTPException(status_code=400, detail="Cannot read the reference uploaded image.")
            
            # Ensure both images have the same size
            if img1.shape != img2.shape:
                logger.warning("Uploaded images have different sizes for logical operation.")
                raise HTTPException(status_code=400, detail="Uploaded images must have the same dimensions.")
        
        # Perform operation
        if operation == 'and':
            processed_img = cv2.bitwise_and(img1, img2)
        elif operation == 'xor':
            processed_img = cv2.bitwise_xor(img1, img2)
        elif operation == 'not':
            processed_img = cv2.bitwise_not(img1)
        else:
            logger.warning(f"Unsupported logical operation: {operation}")
            raise HTTPException(status_code=400, detail="Unsupported logical operation.")
    
        # Encode image to bytes
        encoded_image = encode_image(processed_img)
    
        logger.info(f"Applied {operation} logical operation.")
    
        return StreamingResponse(
            io.BytesIO(encoded_image),
            media_type="image/png"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /logic_operation_rgb endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Endpoint: Histogram Equalization
@app.post("/equalize")
async def equalize_histogram(file: UploadFile = File(...)):
    """
    Perform histogram equalization on the uploaded grayscale image.
    Returns the equalized image.
    """
    try:
        if file is None:
            logger.warning("No file uploaded for histogram equalization.")
            raise HTTPException(status_code=400, detail="File cannot be empty.")
        
        if not file.content_type.startswith('image/'):
            logger.warning(f"Uploaded file is not an image: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image.")
        
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    
        if img is None:
            logger.error("Failed to read uploaded image for histogram equalization.")
            raise HTTPException(status_code=400, detail="Cannot read the uploaded image.")
    
        equalized_img = cv2.equalizeHist(img)
    
        # Encode image to bytes
        encoded_image = encode_image(equalized_img)
    
        logger.info("Applied histogram equalization.")
    
        return StreamingResponse(
            io.BytesIO(encoded_image),
            media_type="image/png"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /equalize endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Endpoint: Histogram Specification
@app.post("/specify_histogram")
async def specify_histogram(file: UploadFile = File(...), ref_file: UploadFile = File(...)):
    """
    Perform histogram specification (matching) using a reference image.
    Returns the specified image.
    """
    try:
        if file is None or ref_file is None:
            logger.warning("Main file or reference file is missing for histogram specification.")
            raise HTTPException(status_code=400, detail="Both main and reference files are required.")
        
        if not file.content_type.startswith('image/') or not ref_file.content_type.startswith('image/'):
            logger.warning("One of the uploaded files is not an image for histogram specification.")
            raise HTTPException(status_code=400, detail="Both files must be images.")
        
        # Read main image
        contents = await file.read()
        img = read_image(contents)
        if img is None:
            logger.error("Failed to read main uploaded image for histogram specification.")
            raise HTTPException(status_code=400, detail="Cannot read the main uploaded image.")
        
        # Read reference image
        ref_contents = await ref_file.read()
        ref_img = read_image(ref_contents)
        if ref_img is None:
            logger.error("Failed to read reference uploaded image for histogram specification.")
            raise HTTPException(status_code=400, detail="Cannot read the reference uploaded image.")
        
        # Perform histogram matching using scikit-image
        try:
            from skimage.exposure import match_histograms
            specified_img = match_histograms(img, ref_img, channel_axis=-1)
            specified_img = np.clip(specified_img, 0, 255).astype('uint8')
        except Exception as e:
            logger.error(f"Failed to perform histogram specification: {e}")
            raise HTTPException(status_code=500, detail="Failed to perform histogram specification.")
        
        # Encode image to bytes
        encoded_image = encode_image(specified_img)
    
        logger.info("Performed histogram specification.")
    
        return StreamingResponse(
            io.BytesIO(encoded_image),
            media_type="image/png"
        )
    except HTTPException as he:
        raise he
    except ImportError:
        logger.error("skimage is not installed. Install it to use histogram specification.")
        raise HTTPException(status_code=500, detail="Required library not installed.")
    except Exception as e:
        logger.error(f"Error in /specify_histogram endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Endpoint: Statistics Calculation
@app.post("/statistics")
async def calculate_statistics(file: UploadFile = File(...)):
    """
    Calculate mean intensity and standard deviation of a grayscale image.
    Returns the statistics.
    """
    try:
        if file is None:
            logger.warning("No file uploaded for statistics calculation.")
            raise HTTPException(status_code=400, detail="File cannot be empty.")
        
        if not file.content_type.startswith('image/'):
            logger.warning(f"Uploaded file is not an image: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image.")
        
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    
        if img is None:
            logger.error("Failed to read uploaded image for statistics calculation.")
            raise HTTPException(status_code=400, detail="Cannot read the uploaded image.")
    
        mean_intensity = float(np.mean(img))
        std_deviation = float(np.std(img))
    
        logger.info(f"Calculated statistics - Mean: {mean_intensity}, Std Dev: {std_deviation}")
    
        return JSONResponse(content={
            "mean_intensity": mean_intensity,
            "standard_deviation": std_deviation
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /statistics endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# Additional endpoints like /add_noise, /remove_noise, etc., follow similar patterns.
