# Deep Learning Inference Performance Benchmark Tool (OpenVINO)
Benchmark Tool (Python / C++) to estimate deep learning inference performance using [openvino](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)

## Setup (Python)
```pip install openvino==2022.1.0```

## Usage (Python)
```python benchmark_onnxruntime.py -i <path_to_xml>```

```
-n: number of inferences
-d: choose inference adapter
```

example:

```python benchmark_onnxruntime.py -i <path_to_xml> -n 100 -d GPU```



   
