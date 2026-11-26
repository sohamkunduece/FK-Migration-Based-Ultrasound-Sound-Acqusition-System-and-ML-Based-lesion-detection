## Python Implementation

This Python implementation follows an architecture similar to the original MATLAB implementation.

MATLAB served as the primary environment for algorithm development, implementation, and testing. Python was subsequently used to deploy and evaluate the processing pipeline in a Python-based environment.

The Python implementation was executed using **Google Colab**.

### Implementation Details

The following Python files correspond to their respective MATLAB implementations:

- `enhancement2.py` — Image enhancement
- `fkmig.py` — Frequency-wavenumber (f-k) migration
- `preprocessing.py` — Ultrasound data preprocessing

These files were stored in Google Drive and imported into the Google Colab environment during execution.

The CNN used for lesion classification was converted from the original MATLAB `.mat` model into the ONNX format. The ONNX model was then used for inference in the Python pipeline.

### Performance Metrics

The following table presents the performance metrics obtained from the RUMC ultrasound processing pipeline.

| Category | Metric | Value |
|---|---|---:|
| Dataset | Remaining Samples | 2,848 |
| Dataset | Number of Channels | 128 |
| Dataset | Transmission Angles | 75 |
| Timing | Processing Latency | 49.228392 sec |
| Timing | Total Execution Time | 49.228392 sec |
| Latency Statistics | Mean Latency | 49.228392 sec |
| Latency Statistics | Minimum Latency | 49.228392 sec |
| Latency Statistics | Maximum Latency | 49.228392 sec |
| Latency Statistics | Latency Jitter | N/A |
| Throughput | Acquisitions per Second | 0.0203 |
| Data Rate | Input RF Size | 53.9062 MB |
| Data Rate | Output Image Size | 1.3906 MB |
| Data Rate | Effective Input Rate | 1.0950 MB/sec |
| Data Rate | Effective Output Rate | 0.0282 MB/sec |
| Classification | Predicted Class | `Hypo_lesion` |

**Note:** Latency jitter was not calculated because only one acquisition was measured. Repeated acquisitions are required to evaluate latency variation.