import tensorrt as trt
import sys
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def printShape(engine):
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            print("input layer: {}, shape is: {} ".format(i, engine.get_binding_shape(i)))
        else:
            print("output layer: {} shape is: {} ".format(i, engine.get_binding_shape(i)))

def onnx2trt(onnx_path, engine_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28 # 256MB

        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        engine = builder.build_cuda_engine(network)

        printShape(engine)

        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

if __name__ == "__main__":
    input_path = "/home/wjd/code/yolov5/runs/train/exp58_second/weights/best.onnx"
    output_path = input_path.replace('.onnx', '.engine')
    onnx2trt(input_path, output_path)
