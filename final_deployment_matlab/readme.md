### The setup uses the same raw data and header format. The output of the CNN is a 3-class based designed. As stated all calculation are done in matlab online.  **1\. Execution Time Performance – MATLAB Online**

| Sl. No. | Processing Module | Execution Time (seconds) |
| ----- | ----- | ----- |
| 1 | Header Loading | 1.597577 |
| 2 | Reconstruction Parameter Setup | 0.001629 |
| 3 | CNN Model Loading | 2.214155 |
| 4 | Ultrasound Data Loading | 6.450509 |
| 5 | Preprocessing | 0.500158 |
| 6 | FK Migration | 6.944481 |
| 7 | Hilbert Envelope Detection | 0.008240 |
| 8 | Normalization | 0.000400 |
| 9 | Gain Enhancement (+10 dB) | 0.000326 |
| 10 | Log Compression | 0.000944 |
| 11 | Four-Stage Enhancement | 0.030783 |
| 12 | Image Quality Evaluation | 0.019245 |
| 13 | CNN Input Preparation | 0.002131 |
| 14 | CNN Classification | 0.815670 |
|  | **Total Execution Time** | **18.586757** |

### **2\. Image Quality Evaluation**

| Sl. No. | Image Quality Metric | FK Migrated Image | Enhanced Image | Difference |
| ----- | ----- | ----- | ----- | ----- |
| 1 | SNR (dB) | 10.2552 | 17.7109 | \+7.4557 |
| 2 | CNR | 0.643920 | 1.059670 | \+0.415750 |
| 3 | ENL | 3.476964 | 74.582703 | \+71.105736 |
| 4 | Mean Intensity | 0.012966 | \-59.186096 | — |
| 5 | Standard Deviation | 0.046948 | 19.183289 | — |
| 6 | Dynamic Range | 1.000000 | 94.744217 | — |

### **3\. Experimental Configuration**

| Parameter | Value |
| ----- | ----- |
| Execution Environment | MATLAB Online |
| Dataset | RUMC Ultrasound Dataset |
| Dataset Class | No Lesion |
| Remaining Samples | 2,848 |
| Number of Channels | 128 |
| Transmission Angles | 75 |
| Sampling Frequency | 31.25 MHz |
| Speed of Sound | 1540 m/s |
| Reconstruction Method | FK Migration |
| Envelope Detection | Hilbert Transform |
| Gain | \+10 dB |
| Enhancement Method | Four-Stage Enhancement |
| Classification Model | Three-Class CNN |

### **4\. CNN Classification Results**

| Sl. No. | Classification Class | Confidence |
| ----- | ----- | ----- |
| 1 | No Lesion | 49.7919% |
| 2 | Hypo Lesion | 31.0038% |
| 3 | Hyper Lesion | 19.2044% |
|  | **Predicted Class** | **No Lesion** |

### **5\. Performance Summary**

| Parameter | Measured Result |
| ----- | ----- |
| Total Execution Time | 18.586757 seconds |
| Ultrasound Data Loading | 6.450509 seconds |
| FK Migration Time | 6.944481 seconds |
| CNN Model Loading | 2.214155 seconds |
| CNN Classification Time | 0.815670 seconds |
| Four-Stage Enhancement | 0.030783 seconds |
| Reported SNR Improvement | \+7.4557 dB |
| Reported CNR Improvement | \+0.415750 |
| Reported ENL Improvement | \+71.105736 |

