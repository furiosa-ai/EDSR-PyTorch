# import tensorrt as trt
# 
# 
# NUM_IMAGES_PER_BATCH = 5
# batchstream = ImageBatchStream(NUM_IMAGES_PER_BATCH, calibration_files)
# 
# Int8_calibrator = EntropyCalibrator(["input_node_name"], batchstream)
# 
# config.set_flag(trt.BuilderFlag.INT8)
# config.int8_calibrator = Int8_calibrator
'''
TensorRT 8.2.3 GA
'''

import tensorrt as trt
import helper
import os
import glob
import argparse

if not os.path.exists( 'onnx_model' ) : os.mkdir( 'onnx_model' )
if not os.path.exists( 'trt_cache' ) : os.mkdir( 'trt_cache' )
if not os.path.exists( 'trt_model' ) : os.mkdir( 'trt_model' )

parser = argparse.ArgumentParser()
parser.add_argument('--dynamic_shape', action='store_true' )
args = parser.parse_args()

# ----- NOTE : you can modify here ------------------------ #
onnx_path = 'onnx_model/EDSR_x4.onnx'
cache_file = 'trt_cache/EDSR_x4.cache.bin'
engine_path = 'trt_model/EDSR_x4.engine'
calibration_files = glob.glob( 'DIV2K/DIV2K_train_LR_bicubic/X4/*.png')
calibration_files = calibration_files[:200]
input_shape = [ 3, 340, 510 ]    # shape without batch axis, CHW
NUM_IMAGES_PER_BATCH = 1
DYNAMIC_SHAPE = args.dynamic_shape
minimum_shape = (1, 3, 231, 510)    # only used when using DYANMIC_SHAPE
optimum_shape = (1, 3, 340, 510)    # only used when using DYANMIC_SHAPE
maximum_shape = (1, 3, 486, 510)    # only used when using DYANMIC_SHAPE

def preprocessor( img ) :
    img = img
    return img

# --------------------------------------------------------- #

def main() :

    # -------------------------------------------
    # Developer Guide 4.1.1 ~ 4.1.2
    # - There is no options to control in this function.
    # -------------------------------------------
    builder, network = helper.create_builder_and_network_from_onnx( onnx_path )

    # -------------------------------------------
    # Developer Guide 4.1.3 Building an Engine
    # - You can specify lots of optimization options here.
    # -------------------------------------------

    config = builder.create_builder_config()

    config.max_workspace_size = 1 << 20 # 1 MiB

    # -------------------------------------------
    # Developer Guide 7.3.2 Calibration Using Python
    # - You can specify lots of optimization options here.
    # - If it is not vision network, you need to replace read_image_chw function in ImageBatchStream.
    # - EntropyCalibrator2 of helper.PythonEntropyCalibrator is recommended for CNN-based network.
    # - EntropyCalibrator2 is recommended for CNN-based networks. If it is not or you need to try other calibrator, you need to inheritance of helper.PythonEntropyCalibrator.
    # -------------------------------------------
    batchstream = helper.ImageBatchStream( NUM_IMAGES_PER_BATCH, calibration_files, input_shape, preprocessor=preprocessor )

    Int8_calibrator = helper.PythonEntropyCalibrator( batchstream, cache_file )

    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = Int8_calibrator

    # config.profiling_verbosity = trt.ProfilingVerbosity.NONE
    # config.profiling_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # Dynamic shape
    #  - You can skip this part if you don't need dynamic shape
    #  - reference :
    #      - https://forums.developer.nvidia.com/t/working-with-dynamic-shape-example/112738
    #      - Developer Guide 2.7, 8

    if DYNAMIC_SHAPE :
        profile = builder.create_optimization_profile()
        profile.set_shape( network.get_input(0).name, minimum_shape, optimum_shape, maximum_shape )
        config.add_optimization_profile(profile)

    # -------------------------------------------
    # Developer Guide 4.1.3 Building an Engine
    # - Create a plan and save it.
    # -------------------------------------------
    print("Building TensorRT engine. This may take few minutes.")
    serialized_engine = builder.build_serialized_network( network, config )
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print( f'Save engine : {engine_path}')

if __name__ == '__main__':
    main()

