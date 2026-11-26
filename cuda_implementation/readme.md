The setup comprises of the cuda deployed over the python setup. Te cude used here is 12.9 version
The metrics showcase a 11 times increase in speed over cpu counterpart.

Note: The function files provided in this folder are all cuda optimized.

The following benchmarks were conducted using Google Colab. The CPU and CUDA implementations process the ultrasound RF data through the same intended processing pipeline.(Note the metrics alggo checks processing only and does account for ouput loading)

| Metric | Python CPU | Python CUDA |
|---|---:|---:|
| Processing Latency | 49.228392 sec | 4.327720 sec |
| Total Execution Time | 49.228392 sec | 4.327720 sec |
| Mean Latency | 49.228392 sec | 4.327720 sec |
| Minimum Latency | 49.228392 sec | 4.327720 sec |
| Maximum Latency | 49.228392 sec | 4.327720 sec |
| Latency Jitter | N/A | N/A |
| Throughput | 0.0203 acquisitions/sec | 0.2311 acquisitions/sec |
| Input RF Size | 53.9062 MB | 56.5248 MB |
| Output Image Size | 1.3906 MB | 1.4582 MB |
| Effective Input Rate | 1.0950 MB/sec | 13.0611 MB/sec |
| Effective Output Rate | 0.0282 MB/sec | 0.3369 MB/sec |
| Speedup | 1× | 11.38× |
| Processing Latency Reduction | — | 91.21% |