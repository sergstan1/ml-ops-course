import fire
import tensorrt as trt


def convert(onnx_path, output_path, fp16=False):
    """
    Convert ONNX model to TensorRT engine

    Args:
        onnx_path: Input ONNX file path
        output_path: Output TensorRT engine path
        fp16: Enable FP16 mode (default: False)
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()

    for idx in range(network.num_inputs):
        input_tensor = network.get_input(idx)
        input_shape = input_tensor.shape

        dynamic_dims = [i for i, dim in enumerate(input_shape) if dim == -1]

        if dynamic_dims:
            min_shape = list(input_shape)
            opt_shape = list(input_shape)
            max_shape = list(input_shape)

            for dim_idx in dynamic_dims:
                min_shape[dim_idx] = 1
                opt_shape[dim_idx] = 224
                max_shape[dim_idx] = 1024

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        else:
            profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)

    config.add_optimization_profile(profile)

    print("Building TensorRT engine...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("Failed to build engine")
        return False

    with open(output_path, "wb") as f:
        f.write(engine)
    print(f"TensorRT engine saved to: {output_path}")
    return True


if __name__ == "__main__":
    fire.Fire(convert)
