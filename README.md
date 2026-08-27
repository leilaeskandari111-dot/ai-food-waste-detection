# AI-Based Food Waste Detection

A reproducible computer-vision proof-of-concept for AI-assisted food-waste detection, inspired by smart-bin systems for food-waste monitoring and reduction.

## Project motivation

Food waste reduction requires not only measuring how much waste is discarded, but also identifying the type and patterns of discarded food.

This project develops a small computer-vision prototype that explores how object detection can be used as one component of an AI-based smart-bin system.

## Research concept

The proposed pipeline is:

Image acquisition  
→ Object detection  
→ Food-item identification  
→ Waste categorisation  
→ Quantitative analysis  
→ Identification of waste patterns  
→ Evaluation of interventions

## Objectives

- Detect food-related objects in images using computer vision.
- Estimate the type and number of detected food items.
- Evaluate model performance using standard classification and detection metrics.
- Explore how computer-vision outputs could support automated food-waste monitoring.
- Provide a reproducible foundation for future integration with a physical smart-bin system.

## Methods

The prototype is implemented in Python and uses modern computer-vision methods for object detection.

The project is designed with reproducibility in mind, including:

- Python-based analysis
- structured source code
- quantitative model evaluation
- visualisation of predictions
- reproducible experiment configuration
- automated testing through GitHub Actions

## Important scope

This repository is a research proof-of-concept rather than a complete physical smart-bin system.

The purpose is to demonstrate the computational pipeline and investigate the feasibility of applying computer vision to food-waste monitoring.

## Computer Vision Pipeline

The current implementation uses a pretrained YOLO object-detection model to demonstrate the food-detection component of the proposed smart-bin workflow.

The pipeline:

1. Loads an input image
2. Detects objects using YOLO
3. Filters food-related classes
4. Counts detected food items
5. Generates an annotated output image
6. Exports detection results as CSV

### Pipeline

```text
Input image
    ↓
YOLO object detection
    ↓
Food-related object filtering
    ↓
Object counting
    ↓
Annotated image + CSV results
```

The current implementation is a proof-of-concept using a pretrained general-purpose object-detection model. It is not presented as a food-waste-specific model trained on a dedicated dataset.

This provides a reproducible starting point for future development of food-waste-specific computer-vision models and smart-bin integration.

## Future development

Possible extensions include:

- training on a dedicated food-waste dataset
- multi-object detection
- food-waste quantity estimation
- temporal analysis of waste patterns
- integration with a camera-equipped smart bin
- deployment on edge hardware
- analysis of interventions designed to reduce food waste

## Reproducibility

The complete analysis pipeline, dependencies and experimental outputs will be documented in this repository.

## Author

Leila Eskandari
