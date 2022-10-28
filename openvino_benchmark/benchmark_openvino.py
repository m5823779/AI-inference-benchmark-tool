import os
import time
import argparse
import numpy as np
from openvino.inference_engine import ie_api as ie

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_model", required=True, help="Input model .onnx or .xml (OpenVINO IR model)")
parser.add_argument('-d', '--device', default='CPU', help='select inference adapter (CPU, GPU, VPU etc.)')
parser.add_argument('-n', '--num_infers', default=100, type=int, help='number of inferences')
args = parser.parse_args()

num_infers = args.num_infers
input_model = args.input_model

# Timer
t = 0.0

# Load IE
core = ie.IECore()

root, extension = os.path.splitext(input_model)
if extension == ".xml":
    bin = root + ".bin"
    net = core.read_network(model=input_model, weights=bin)
elif extension == ".onnx":
    net = core.read_network(model=input_model)
else:
    print(f"Cant support {extension} format, please input .xml or .onnx model.")
    print("Exit ...")
    exit()

exec_net = core.load_network(net, device_name=args.device, num_requests=1)

input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))
input_info = exec_net.input_info
assert (len(input_info) == 1)
input_name = next(iter(input_info))

# Get model input shape
input_shape = tuple(input_info[input_name].input_data.shape)
print("Network input size: {}".format(input_shape))

# Create dummy input
sample = np.ndarray((input_shape))

for i in range(num_infers):
    start = time.perf_counter()
    # Inference
    req = exec_net.requests[0]
    req.infer(inputs={input_blob: sample})
    end = (time.perf_counter() - start) * 1000
    t += end
    print('\rInference times: %.2f ms' % end, end='')
t /= num_infers
print(f"\nAverage inference time: {t:.2f}ms\n")