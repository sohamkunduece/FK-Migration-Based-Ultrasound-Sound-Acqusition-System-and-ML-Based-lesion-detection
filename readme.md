# **FK-Migration-Based Ultrasound Acquisition System and ML-Based Lesion Detection**

A CUDA- and Holoscan-based ultrasound image reconstruction and lesion classification pipeline developed for the MathWorks MATLAB and Simulink Challenge Project: **Real-Time Acceleration for Medical Image Processing**.

## **Objective**

The objective of this project is to develop a Holoscan-based ultrasound processing system that receives raw radio-frequency (RF) acquisition data along with probe configuration information, reconstructs and enhances the ultrasound image, and subsequently applies a machine learning model to classify the lesion present in the image region.

The system demonstrates a complete medical imaging workflow, from raw RF data processing to image reconstruction, enhancement, and CNN-based predictive analysis.

## **Introduction**

This project was developed in response to the MathWorks and Simulink Challenge Project, *Real-Time Acceleration for Medical Image Processing*.

The proposed approach follows a modular medical imaging pipeline in which raw ultrasound RF data is first reconstructed into an image, enhanced using image-processing techniques, and then passed to an AI-based classification interface.

A Convolutional Neural Network (CNN) is used for lesion classification in this demonstration.

The project explores multiple implementations of the processing pipeline, including MATLAB, Python, CUDA, and NVIDIA Holoscan.

## **Processing Pipeline**

The processing pipeline follows the sequence below:

```
Raw RF Data + Probe Header Information
                 |
                 v
       RF Preprocessing
                 |
                 v
       Stolt's F-K Migration
                 |
                 v
       Image Reconstruction
                 |
                 v
       Hilbert Transform
                 |
                 v
       Image Enhancement
                 |
                 v
       CNN-Based Classification
                 |
                 v
      Enhanced Image + Prediction
```

### **1\. RF Preprocessing**

The preprocessing stage prepares the acquired RF data for reconstruction. It includes:

* Correcting and organizing the input dataset.  
* Handling probe steering-angle information.  
* Identifying and removing or correcting corrupted portions of RF data.  
* Preparing the RF data according to the acquisition configuration.

### **2\. Image Reconstruction**

Stolt's frequency-wavenumber (F-K) migration is used to reconstruct the ultrasound image from the plane-wave RF data.

The F-K migration implementation is based on the reference repository listed in the References section.

### **3\. Envelope Detection**

A Hilbert transform is applied to obtain the envelope of the reconstructed RF signal.

### **4\. Image Enhancement**

The reconstructed image is enhanced using image-processing filters to improve its visual quality and support subsequent analysis.

### **5\. CNN-Based Lesion Classification**

The enhanced ultrasound image is passed to a CNN model for predictive analysis.

The model produces a classification result and the corresponding prediction information.

## **Dataset**

The primary reconstruction dataset used in this project is the **75-Angle Phantom Dataset**, containing 240 files.

The reconstructed images were compared with the corresponding images presented in the associated research paper. The reconstruction results showed effective agreement with the reference images.

Due to limitations in available raw RF data and computational resources, 40 files out of the 240 available files were used for the CNN training workflow.

The project also includes additional datasets for testing and evaluation.

### **Dataset Evaluation**

Two additional datasets were used to evaluate different aspects of the pipeline:

| Dataset | Purpose |
| ----- | ----- |
| OASBUD | Evaluation of image enhancement performance |
| PICMUS | Testing the reconstruction pipeline on another raw ultrasound dataset |

The project repository contains the datasets, test data, reconstructed images, and other relevant implementation files.

## **Machine Learning Pipeline Approach**

During development, two approaches were considered for connecting the ultrasound processing pipeline to the CNN.

### **Type 1: Direct Input to the ML Model**

In this approach, the processed input data is passed directly to the machine learning model without creating an intermediate image file.

Advantages:

* Lower intermediate storage requirements.  
* Reduced file-writing overhead.  
* Potentially faster processing.  
* Avoidance of additional image conversion steps.

However, the model may be more susceptible to overfitting when the available training dataset is limited.

In this project, 40 files out of the 240 available files were used for the CNN training workflow due to storage and computational limitations.

### **Type 2: Intermediate Image-Based Input**

In this approach, an intermediate image representation, such as raw image data or a PNG file, is generated and then supplied to the ML model.

Although this approach can provide a more explicit image-based training context, it introduces additional storage requirements, file operations, and processing overhead.

It may also produce inconsistent predictions if the intermediate image representation changes between training and inference.

For these reasons, the intermediate image-based approach was not used in the final implementation.

### **Intermediate Reconstruction Data**

The repository includes a file named `complete.reconstructed_image` to provide context regarding the intermediate data generated during the reconstruction pipeline.

This file illustrates how the raw RF data is transformed into a reconstructed image representation.

## **F-K Migration Function**

The F-K migration function requires probe-specific acquisition parameters. If the acquisition probe or configuration changes, the corresponding variables must be updated.

| Parameter | Description |
| ----- | ----- |
| `param.x` | Defines the lateral (x-axis) coordinates of the reconstructed image |
| `param.z` | Defines the axial (z-axis) coordinates of the reconstructed image |
| `Fs` | Sampling frequency of the acquisition probe |
| `P` | Convergence parameter; generally kept at 0 in this implementation |
| `Txangle` | Transmission steering angles of the probe, expressed in degrees |
| `C` | Speed of sound in the medium |
| `Passband` | Bandpass frequency range associated with the probe |
| `Pitch` | Element spacing of the ultrasound probe |
| `Ncol` | Number of acquisition channels or probe columns |
| `T0` | Time-zero correction used to compensate for delayed beam firing; set to 0 in the current configuration |

**Important:** Probe configuration and acquisition parameters must match the RF dataset. Incorrect parameter values may lead to reconstruction artifacts or inaccurate image geometry.

## **Repository Structure**

The repository is organized into separate folders for acquisition, reconstruction, machine learning, deployment, and testing.

| Folder | Description |
| ----- | ----- |
| `Acquisition_setup` | Acquisition configuration and conversion of raw RF data into reconstructed images |
| `Cnn_model_and_details` | CNN development, model details, and classification information |
| `Final_deployment_matlab` | Complete MATLAB implementation of the processing pipeline |
| `Python_implementation` | Python implementation of the MATLAB processing pipeline |
| `Cuda_implementation` | CUDA-optimized Python implementation |
| `Holoscan` | Headless NVIDIA Holoscan deployment |
| `Test-1` | MATLAB pipeline testing under different image conditions |
| `Test-2` | MATLAB pipeline testing using another probe or dataset configuration |

Each implementation folder contains its own README with relevant setup instructions, execution details, metrics, and supporting information.

## **Performance Observations**

The project was evaluated using MATLAB, Python, CUDA, and Holoscan implementations.

### **Processing Time**

| Implementation | Reported Processing Time |
| ----- | ----- |
| MATLAB implementation | Approximately 9 seconds |
| Python implementation | Approximately 37 seconds |
| CUDA implementation | Approximately 5 seconds |
| Holoscan implementation | Approximately 8 seconds |

The CUDA implementation reduced the processing time compared with the Python implementation.

The Holoscan implementation demonstrated successful pipeline integration, although its measured processing time was higher than the direct CUDA implementation in the current configuration.

### **Batch Processing**

A batch experiment was conducted using 40 raw ultrasound files.

| Metric | Result |
| ----- | ----- |
| Number of input files | 40 |
| Approximate input file size | 55 MB per file |
| Total input data | Approximately 2.2 GB |
| Total batch execution time | 388.5 seconds |
| Output images | 40 PNG files |
| CNN prediction outputs | 40 text files |

The batch experiment involved reading the input files from Google Drive, processing the ultrasound data, and writing the enhanced images and CNN prediction results back to Google Drive.

The measured execution time therefore includes data transfer and cloud-storage overhead in addition to computation.

Some individual processing stages were observed to execute in sub-second or millisecond time ranges.

### **Performance Interpretation**

The reported metrics represent a cloud-based development and testing environment rather than a native hardware benchmark.

The results demonstrate the feasibility of GPU-accelerated ultrasound processing and Holoscan integration. Further optimization of data movement, pipeline scheduling, and storage operations may improve overall latency.

The current implementation should be considered a GPU-accelerated prototype rather than a validated real-time clinical ultrasound system.

## **Deployment Environment**

The project was developed and tested using the following environments:

| Component | Environment |
| ----- | ----- |
| MATLAB development | MATLAB Online |
| Python implementation | Google Colab |
| CUDA implementation | Google Colab NVIDIA GPU |
| GPU | NVIDIA Tesla T4 |
| Holoscan deployment | Google Colab, headless |
| CUDA toolkit | CUDA 12.9 |
| Holoscan | 4.6.0 |
| Python | Google Colab Python environment |

The MATLAB implementation and CNN model were developed in MATLAB Online.

Python, CUDA, and Holoscan deployment experiments were conducted using Google Colab CPU and NVIDIA Tesla T4 GPU resources.

### **GPU Coder Limitation**

MATLAB GPU Coder was unavailable in the development environment. Therefore, the CUDA acceleration and Holoscan deployment were implemented and evaluated using the available Python and CUDA environment.

This alternative implementation allowed the project to explore GPU acceleration and Holoscan integration without relying on MATLAB GPU Coder.

## **Generative AI Usage**

Generative AI tools, including Claude by Anthropic and chatgpt by Openai, were used during the project to assist with debugging, pipeline integration, and documentation drafting.

All technical decisions and reported results were reviewed and verified by the author. Reported performance metrics were obtained from the corresponding committed scripts and output files.

## **References**

1. **Stolt's F-K Migration**  
   [Plane Wave Ultrasound Stolt F-K Migration](https://github.com/rehmanali1994/Plane_Wave_Ultrasound_Stolt_F-K_Migration.github.io/tree/master/MATLAB)  
2. **75-Angle Phantom Dataset**  
   [Zenodo Dataset](https://zenodo.org/records/7986407)  
3. **75-Angle Dataset Research Paper**  
   [Research Paper on arXiv](https://arxiv.org/abs/2404.14188)  
4. **OASBUD Dataset**  
   [OASBUD Research Publication](https://pubmed.ncbi.nlm.nih.gov/28859252/)  
5. **PICMUS Dataset**  
   [IEEE IUS 2016 Ultrasound Challenge](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/home)

## **Author**

**Soham Kundu**  
B.Tech, Electronics and Communication Engineering  
Meghnad Saha Institute of Technology  
Kolkata, West Bengal, India

