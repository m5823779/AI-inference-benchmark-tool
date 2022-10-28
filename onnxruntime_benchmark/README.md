# Deep Learning Inference Performance Benchmark Tool (ONNX runtime)
Benchmark Tool (Python) to estimate deep learning inference performance using [onnx runtime](https://onnxruntime.ai/)

## Support Hardware Acceleration (Optional)
- [x] CPU (default)
- [x] Direct ML  
- [x] OpenVINO
- [x] TensorRT

## Setup
```pip install -r requirements.txt```

for different hard acceleration (optional)

Direct ML: ```pip install onnxruntime-directml```

TensorRT: ```pip install onnxruntime-gpu```

## Usage
list all available execution providers:

```python benchmark_onnxruntime.py -p```

```python benchmark_onnxruntime.py -i <path_to_onnx_model>```

```
-w: set model input height and width if model input shape is dyanmic
-d: choose inference hardware adapter id
-n: number of inferences
-p: list all available execution providers
-e: setting execution providers
```

example:

```python benchmark_onnxruntime.py -i <path_to_onnx_model> -w 256 -d 1 -n 100 -e DmlExecutionProvider```

```python benchmark_onnxruntime.py -i <path_to_onnx_model> -w 256 -d 0 -n 100 -e CUDAExecutionProvider```


   
