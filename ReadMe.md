# Effect of Input Modality on Mathematical Reasoning in Vision-Language Models

## Overview

This project studies how different input modalities affect mathematical reasoning performance in vision-language models. Using the GSM8K dataset of grade-school math word problems, we evaluate how a multimodal model performs under different input formats while keeping the model, dataset, and evaluation pipeline fixed.

We use the Qwen2-VL-7B-Instruct model from the Qwen family to compare reasoning performance across text-only, image-based, and multimodal settings.

## Objectives

- Compare mathematical reasoning performance across different input modalities
- Evaluate the impact of vision encoders on GSM8K reasoning tasks
- Study robustness to different visual representations (clean images vs screenshots)
- Analyze multimodal fusion (text + image) effects
- Perform error classification across conditions

## Note: Screenshot experimentation will be implemented at a later time.

## Experimental Setup

### Dataset
- GSM8K (Grade School Math Word Problems)
- 100 samples from the test split

### Input Conditions

1. Text-only  
   Raw GSM8K question as text

2. Rendered Image  
   Questions converted into clean images using PIL rendering

3. Screenshot Input  (To be implemented)
   Real-world screenshot-style images with visual noise

4. Text + Image Fusion  
   Both image and original text provided to the model

### Model
- Qwen2-VL-7B-Instruct

### Hardware
- Google Colab
- NVIDIA T4 GPU

## How to Run (Google Colab)

1. Open the notebook:
   VLM_GSM8K_Benchmarking.ipynb

2. Enable GPU:
   Runtime → Change runtime type → GPU (T4)

3. Install dependencies:
   pip install torch transformers datasets pillow tqdm pandas

4. Run the notebook sequentially:
   - Load GSM8K dataset
   - Load Qwen2-VL model
   - Run experiments for all conditions:
     - Text-only
     - Rendered image
     - Screenshot input
     - Text + image fusion

5. View results:
   The notebook computes accuracy, error types, and saves results automatically.

## Evaluation Method

We evaluate model outputs using:
- Exact numeric matching against GSM8K ground truth
- Extraction of final numeric answer from model output
- Error classification into:
  - arithmetic errors
  - reasoning errors
  - vision/OCR errors
  - missing or invalid outputs


## Key Findings

- Text-only reasoning provides a strong baseline performance
- Vision input can improve or degrade performance depending on image quality
- Screenshot inputs test real-world robustness of vision encoders
- Multimodal fusion does not guarantee consistent improvement over text-only reasoning
- Error patterns differ significantly across modalities

## Outputs Generated

- text_only_results.csv
- rendered_image_results.csv
- screenshot_results.csv
- fusion_results.csv
- disagreements.csv

## Limitations

- Small evaluation size (100 samples)
- Single main model (Qwen2-VL-7B-Instruct)
- Limited by T4 GPU constraints
- Some variation due to nondeterministic generation

## Future Work

- Scale to full GSM8K dataset
- Evaluate more vision-language models
- Improve multimodal fusion prompting strategies
- Test additional reasoning datasets (SVAMP, AQuA-RAT)
- Optimize inference efficiency for low-resource GPUs

## Author

Rodela Ghosh  
University of South Florida (USF)

Aviral Gupta 
University of South Florida (USF)

## Acknowledgements

- GSM8K dataset creators
- Hugging Face Transformers library
- Qwen model contributors