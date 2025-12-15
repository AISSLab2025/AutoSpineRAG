import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from model_utils import ensembleInference, preProcessImage
from axial_utils import applyAxialMeasurements
from fastapi import FastAPI, File, UploadFile,  Form
from fastapi.responses import JSONResponse, Response
import numpy as np
import cv2
from PIL import Image
import io
import base64
from sagittal_utils import applyAnglesMeasurements, applyDistanceMeasurementsOnImage, applySpinalHeightMeasurements, applySpondylolisithesisClassification, applyLumbarLordosisClassification, convert_to_builtin_types, convertToCategoricalMasks, getContours, getIVDAndPEClass
import torch
import wandb

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello World"}
#
#################### 1. Distance Measurement ####################
@app.post("/process_segmentation/")
async def process_segmentation(
    required_structure_measurements: str = Form(...),  # Form parameter
    pixel_spacing: float = Form(...),  # Form parameter
    mask_file: UploadFile = File(...),  # File parameter
    image_file: UploadFile = File(...),  # File parameter
):
    # Load the mask and image
    print("\nrequired_structure_measurements", required_structure_measurements)
    mask = np.array(Image.open(io.BytesIO(await mask_file.read())))
    image = np.array(Image.open(io.BytesIO(await image_file.read())))
    # print("readed mask :", mask)
    # Convert mask to categorical values
    mask_classes = convertToCategoricalMasks(mask)
    # print("mask_classes", mask_classes)
    
    # Get filtered mask (IVD, Vertebrae, Sacrum)
    filtered_mask = getIVDAndPEClass(mask_classes)
    # print("filtered_mask", filtered_mask)
    # Extract contours
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])
    

    
    # Draw measurements
    overlay_image, measurements_dict = applyDistanceMeasurementsOnImage(image.copy(), filtered_mask, contours_dict, pixel_spacing, required_structure_measurements, skeletonization=False)
    # Convert measurements_dict to built-in Python types
    measurements_dict_cleaned = convert_to_builtin_types(measurements_dict)
    # print("instance measurements_dict", measurements_dict)
    # Encode overlay image to PNG
    _, img_encoded = cv2.imencode(".png", overlay_image)
        # Convert image to Base64
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
    print("measurements_dict", measurements_dict_cleaned)
    return JSONResponse(content={
        "measurements": measurements_dict_cleaned,
        "image": img_base64
        # "image": img_encoded.tobytes().hex()  # Convert to hex string for JSON
    })

#################### 2. Angle Measurement ####################
@app.post("/apply_angle_measurements/")
async def apply_angle_measurements(
    image_file: UploadFile = File(...),  # Image file
    mask_file: UploadFile = File(...),  # Mask file
    # pixel_spacing: float = Form(...),  # Pixel spacing
    required_angle_measurements: str  = Form("LLA,LSA")  # Default to LLA and LSA
):
    # Load the image and mask
    image = np.array(Image.open(io.BytesIO(await image_file.read())))
    mask = np.array(Image.open(io.BytesIO(await mask_file.read())))

    # Placeholder functions for contour extraction and processing (Assuming they're already implemented)
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])  # Example structure names

    # Apply angle measurements
    spinal_angle_key_points = {
        "L1": {"top_line_points": []},
        "S1": {"top_line_points": []},
        "L5": {"bottom_line_points": []},
        "LLA": None,
        "LSA": None,
    }

    # Define colors and labels (hardcoded for now)
    contour_colors = {'IVD': (255, 0, 0), 'Vertebrae': (0, 255, 0), 'Sacrum': (0, 0, 255)}
    contours_dict = getContours(filtered_mask,  structure_names =[   "IVD", 'Vertebrae', "Sacrum"])
    # Apply angle calculations
    measurements_dict, overlay_image = applyAnglesMeasurements(
        image, mask, contours_dict,  required_angle_measurements.split(","),
        contour_colors=contour_colors
    )

    # Convert the image to Base64
    _, img_encoded = cv2.imencode(".png", overlay_image)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

    return JSONResponse(content={
        "measurements": measurements_dict,
        "image": img_base64  # Base64 encoded image with overlay
    })

##################### 3. Spinal height measurement #####################
@app.post("/apply_spinal_height_measurements/")
async def apply_spinal_height_measurements(
    pixel_spacing: float = Form(...),
    mask_file: UploadFile = File(...),
    image_file: UploadFile = File(...)
):

    image = np.array(Image.open(io.BytesIO(await image_file.read())))
    mask = np.array(Image.open(io.BytesIO(await mask_file.read())))
    
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])  # Example structure names

    # Call spinal height function
    measurements_dict, overlay_image = applySpinalHeightMeasurements(image, filtered_mask, contours_dict, pixel_spacing)
    
    print(measurements_dict)
    # Convert processed image to base64 for response
    _, buffer = cv2.imencode(".png", overlay_image)
    processed_image_b64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        "measurements": convert_to_builtin_types(measurements_dict),
        "image": processed_image_b64,
    }

###################### 4. Spondylolisthesis Measurement ######################
@app.post("/apply_spinal_height_measurements/")
async def apply_spinal_height_measurements(
    pixel_spacing: float = Form(...),
    mask_file: UploadFile = File(...),
    image_file: UploadFile = File(...)
):

    image = np.array(Image.open(io.BytesIO(await image_file.read())))
    mask = np.array(Image.open(io.BytesIO(await mask_file.read())))
    
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])  # Example structure names

    # Call spinal height function
    measurements_dict, overlay_image = applySpinalHeightMeasurements(image, filtered_mask, contours_dict, pixel_spacing)
    
    print(measurements_dict)
    # Convert processed image to base64 for response
    _, buffer = cv2.imencode(".png", overlay_image)
    processed_image_b64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        "measurements": convert_to_builtin_types(measurements_dict),
        "image": processed_image_b64,
    }

@app.post("/spondylolisithesis_classification/")
async def classify_spondlylisithesis(
    mask_file: UploadFile = File(...),
    image_file: UploadFile = File(...)
):

    image = np.array(Image.open(io.BytesIO(await image_file.read())))
    mask = np.array(Image.open(io.BytesIO(await mask_file.read())))
    
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])  # Example structure names

    # Call spinal height function
    classification, overlay_image = applySpondylolisithesisClassification(image, filtered_mask, contours_dict)
    
    print(classification)
    # Convert processed image to base64 for response
    _, buffer = cv2.imencode(".png", overlay_image)
    processed_image_b64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        "classification": convert_to_builtin_types(classification),
        "image": processed_image_b64,
    }

@app.post("/lumbar_lordosis_classification/")
async def classify_lumbar_lordosis(
    pixel_spacing: float = Form(...),
    mask_file: UploadFile = File(...),
    image_file: UploadFile = File(...)
):

    image = np.array(Image.open(io.BytesIO(await image_file.read())))
    mask = np.array(Image.open(io.BytesIO(await mask_file.read())))
    
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])
    
    classification, overlay_image = applyLumbarLordosisClassification(image, filtered_mask, contours_dict, pixel_spacing)

    print(classification)
    # Convert processed image to base64 for response
    _, buffer = cv2.imencode(".png", overlay_image) 
    processed_image_b64 = base64.b64encode(buffer).decode("utf-8")
    return {
        "classification": convert_to_builtin_types(classification),
        "image": processed_image_b64,
    }

################################################################
################ Axial Measurement #############################
@app.post("/axial_measurements/")
async def classify_lumbar_lordosis(
    pixel_spacing: float = Form(...),
    mask_file: UploadFile = File(...),
    image_file: UploadFile = File(...)
):

    image = np.array(Image.open(io.BytesIO(await image_file.read())))
    mask = np.array(Image.open(io.BytesIO(await mask_file.read())))
    
    measuremetns_info, overlay_image = applyAxialMeasurements(image, mask, pixel_spacing)
    print(measuremetns_info)
    # Convert processed image to base64 for response
    _, buffer = cv2.imencode(".png", overlay_image) 
    processed_image_b64 = base64.b64encode(buffer).decode("utf-8")
    return {
        "measurements": convert_to_builtin_types(measuremetns_info),
        "image": processed_image_b64,
    }

###############################################################
################ Model Inference ###############################

@app.post("/model_inference_{view}/")
async def classify_lumbar_lordosis(
    view: str,
    image_file: UploadFile = File(...)
):
    print("view", view)
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VIEWE_PLANE = view
    image_byte = io.BytesIO(await image_file.read())
    preprocessed_image = preProcessImage(image_byte)
    preprocessed_image = preprocessed_image.to(device)
    pred_mask = ensembleInference(preprocessed_image, VIEWE_PLANE)

    print("shape of pred_mask", pred_mask.shape)
    # Convert processed image to base64 for response
    _, buffer = cv2.imencode(".png", pred_mask) 
    pred_mask = base64.b64encode(buffer).decode("utf-8")
    return {
        "pred_mask": pred_mask,
    }
