import tensorrt as trt

# ----- NOTE : you can modify here ------------------------ #
engine_path = 'trt_model/EDSR_x4.engine'

# --------------------------------------------------------- #

def main() :

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open( engine_path, 'rb' ) as f :
        serialized_engine = f.read()

    engine = runtime.deserialize_cuda_engine( serialized_engine )
    context = engine.create_execution_context()

    # engine.profiling_verbosity have to be configured while creating engine

    inspector = engine.create_engine_inspector()
    inspector.execution_context = context # OPTIONAL
    # print(inspector.get_layer_information(0, trt.LayerInformationFormat.JSON)) # Print the information of the first layer in the engine.
    print(inspector.get_engine_information( trt.LayerInformationFormat.JSON)) # Print the information of the entire engine.

if __name__ == '__main__':
    main()

