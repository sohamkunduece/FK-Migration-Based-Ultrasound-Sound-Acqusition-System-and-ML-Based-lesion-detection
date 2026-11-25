### CUDA and Holoscan Performance Evaluation

The given functions are CUDA-optimized to accelerate the ultrasound image processing pipeline. However, integrating Holoscan increased the processing time to approximately **3 seconds**.

A batch test was conducted using **40 raw ultrasound data files**, with a total execution time of **388.5 seconds**. Each input file was approximately **55 MB** in size.

It is important to note that the batch test involved writing the generated outputs to Google Drive. For the 40 input files, the system generated:
- 40 CNN classification result text files.
- 40 final enhanced ultrasound images in PNG format.

Additionally, the execution was performed in Google Colab, where internet connectivity and Google Drive transfer overhead may contribute to the overall processing time. Therefore, the measured batch execution time includes not only computation but also data transfer and output-writing overhead.

Consequently, the measured performance should be interpreted as a cloud-based end-to-end benchmark rather than a purely computational benchmark. Execution on a native device with local storage and direct GPU access may reduce the additional overhead associated with cloud storage and internet connectivity.

### Performance Metrics

| Category | Metric | Result |
|---|---|---|
| GPU | GPU Model | NVIDIA Tesla T4 |
| CUDA | CUDA Test | 45.0 |
| CuPy | CuPy Version | 13.6.0 |
| ONNX | Execution Providers | TensorRT, CUDA, CPU |
| ONNX | Input Shape | `['BatchSize', 1, 2848, 128]` |
| ONNX | Output Shape | `['BatchSize', 3]` |
| Preprocessing | Remaining Samples | 2848 |
| Preprocessing | Number of Channels | 128 |
| Preprocessing | Transmission Angles | 75 |
| Preprocessing | Output Device | CUDA GPU |
| Preprocessing | Output Data Type | float64 |
| Preprocessing | Output Shape | `(2848, 128, 75)` |
| FK Migration | Output Device | CUDA GPU |
| FK Migration | Output Data Type | float64 |
| FK Migration | Output Shape | `(2848, 128)` |
| Processing Latency | Average | 7.328978 sec |
| End-to-End Latency | Average | 7.889467 sec |
| CNN Inference | Average Inference Latency | 0.208021 sec |
| Throughput | Samples Processed | 1 |
| Throughput | Samples per Second | 0.1268 |
| Throughput | Samples per Minute | 7.61 |
| Holoscan | Processing Time | Approximately 7 sec |
| Batch Processing | Number of Input Files | 40 |
| Batch Processing | Total Execution Time | 388.5 sec |
| Input Data | Approximate Size per File | 55 MB |
| Output Data | CNN Result Files | 40 text files |
| Output Data | Enhanced Images | 40 PNG files |
| Storage | Output Destination | Google Drive |
| Environment | Platform | Google Colab |
