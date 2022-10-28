import argparse
import time
import onnx
import onnxruntime
import numpy as np


def onnx_set_input_shape(onnx_proto, input_wh):
    # replace input/output dynamic shape to static shape for DmlExecutionProvider for faster inference
    #   reference - https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#performance-tuning
    for d in onnx_proto.graph.input[0].type.tensor_type.shape.dim:
        if d.dim_param:
            d.dim_value = input_wh

    for d in onnx_proto.graph.output[0].type.tensor_type.shape.dim:
        if d.dim_param:
            d.dim_value = input_wh
    model_payload = onnx_proto.SerializeToString()
    return model_payload


def benchmark(model_path, execution_provider, num_infers, input_wh, device_id):
    execution_providers = [execution_provider]

    session_options = onnxruntime.SessionOptions()
    if execution_provider == 'DmlExecutionProvider':
        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_pattern = False

    onnx_proto = onnx.load(model_path)
    model_payload = onnx_set_input_shape(onnx_proto, input_wh)

    # session_options.add_free_dimension_override_by_name('input_cx', 256)
    # session_options.add_free_dimension_override_by_name('input_cy', 256)

    # session = onnxruntime.InferenceSession(model_path, session_options, providers=execution_providers)
    session = onnxruntime.InferenceSession(model_payload, session_options, providers=execution_providers)

    if execution_provider == 'DmlExecutionProvider':
        session.set_providers(execution_providers, [{'device_id': device_id}])
    if execution_provider == 'OpenVINOExecutionProvider':
        # session.set_providers(['OpenVINOExecutionProvider'], [{'device_type': 'GPU_FP32'}])
        session.set_providers(['OpenVINOExecutionProvider'], [{'device_type': 'CPU_FP32'}])

    input_node = session.get_inputs()[0]
    input_name = input_node.name
    input_shape = input_node.shape
    input_type = np.float16 if "float16" in input_node.type else np.float32

    input_shape = [input_wh if (item == 'input_cx' or item == 'input_cy') else item for item in input_shape]
    print("input_shape: {}".format(input_shape))
    
    total = 0.0
    runs = num_infers
    input_data = np.random.rand(*input_shape).astype(input_type)

    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model", required=False, help="input model")
    parser.add_argument('-w', '--input_wh', default=256, type=int, help='input node shape width & height')
    parser.add_argument('-d', '--device', default=0, type=int, help='Choose hardware device')
    parser.add_argument('-n', '--num_infers', default=30, type=int,help='number of inferences')
    parser.add_argument('-p', '--exec_list', required=False, action='store_true', help='List execution providers.')
    parser.add_argument('-e', '--exec_provider', default='CPUExecutionProvider',
                        help='Execution Provider (CPUExecutionProvider, DmlExecutionProvider, OpenVINOExecutionProvider, etc.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if (args.exec_list):
        # print all possible providers supported by this version of onnxruntime
        all_eps = onnxruntime.get_all_providers()
        print(f"All execution providers:")
        for ep in all_eps:
            print(' -', ep)

        # print available possible providers
        avail_eps = onnxruntime.get_available_providers()
        print("\nAvailable execution providers:")
        for ep in avail_eps:
            print(' -', ep)
    else:
        print('benchmarking model...')
        benchmark(args.input_model, args.exec_provider, args.num_infers, args.input_wh, args.device)


if __name__ == '__main__':
    main()