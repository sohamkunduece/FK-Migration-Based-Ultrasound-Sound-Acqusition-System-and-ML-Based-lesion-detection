The test is done on PICMUS . Since the File they gave was in hdf5, we had to convert them to .mat files. The images from both our pipeline and their provided reference are provided here. It is interesting to note, even if the pipeline format remained unchanged, 2 major changes were observed this pipeline accounted for.  
1\. The probe data was provided in fs,ncol,Tx, pitch and sound speed. On wrong data, the system fails to produce the correct image.  
2\. Due to a single mat file containing both rf and probe details, loading wasting time is reduced by 2 seconds.

For all this only 4 or 5 lines changes altogether in the top file and preprocessing which has been marked.

 Overall the statistics shows:

### **Dataset Configuration**

* Transmission angles: 75  
* Number of channels: 128  
* Remaining samples after preprocessing: 3232  
* Sampling frequency: 20.832 MHz  
* Speed of sound: 1540 m/s  
* Reconstruction method: Multi-angle F-k migration  
* Gain enhancement: \+10 dB

### **Computational Performance**

| Processing Stage | Execution Time |
| ----- | ----- |
| Data loading | 12.682585 s |
| Header loading | 0.000043 s |
| Parameter setup | 0.001885 s |
| Preprocessing | 0.467100 s |
| F-k migration | 6.366728 s |
| Envelope detection | 0.011574 s |
| Normalization | 0.000670 s |
| Gain enhancement | 0.000419 s |
| Log compression | 0.001155 s |
| Four-stage enhancement | 0.035514 s |
| Image quality evaluation | 0.008770 s |
| **Total execution time** | **19.577317 s** |

### **Acquisition Processing Performance**

The processing latency excludes initial data loading, header loading, parameter setup and image quality evaluation.

| Metric | Result |
| ----- | ----- |
| Mean latency | 6.883160 s |
| Minimum latency | 6.883160 s |
| Maximum latency | 6.883160 s |
| Estimated throughput | 0.1453 acquisitions/s |

Throughput is estimated as the reciprocal of the single-acquisition processing latency. These measurements represent one acquisition and are not yet multi-acquisition streaming statistics.

### **Image Quality Metrics**

| Metric | F-k Migrated | Enhanced |
| ----- | ----- | ----- |
| SNR (dB) | 16.0758 | 17.2188 |
| CNR | 1.617774 | 2.362160 |
| ENL | 3.261026 | 5.227164 |
| Mean intensity | 0.008903 | 0.008747 |
| Standard deviation | 0.025399 | 0.023825 |
| Dynamic range | 0.999999 | 0.866676 |

### **Image Quality Improvement**

| Metric | Difference |
| ----- | ----- |
| SNR | \+1.1431 dB |
| CNR | \+0.744385 |
| ENL | \+1.966139 |

The enhancement stage produced improvements in the selected ROI-based SNR, CNR and ENL measurements.

These results indicate improved contrast and increased homogeneity in the selected background region. Further evaluation of lesion boundaries and fine tissue details is required to establish that smoothing does not remove diagnostically useful information.

