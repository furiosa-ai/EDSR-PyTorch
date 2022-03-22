import os
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import common

def create_builder_and_network_from_onnx( onnx_path ) :

    # Developer Guide 4.1 The Build Phase

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # Developer Guide 4.1.1 Creating a Network Definition in Python

    # Explicit batch mode is mandatory for ONNX parser
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Developer Guide 4.1.2 Importing a Model using the ONNX Parser
    parser = trt.OnnxParser( network, logger )

    success = parser.parse_from_file( onnx_path )
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        assert False, 'Onnx Parsing error.'

    return builder, network


class ImageBatchStream:
    '''

    Reference :
        - kevin's edge_ai yolo code
        - https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/

    NOTE :
        - If it is not vision network, you need to replace read_image_chw by yourself.
        - Calibration data has unified shape ( height, width ) by Image resizing. Even though when using dynamic shape, we use unified shape for convenience
    '''
    def __init__(self, batch_size, calibration_files, input_shape, preprocessor=None):
        self.channels, self.height, self.width = input_shape

        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // batch_size) + \
                           (1 if (len(calibration_files) % batch_size) \
                            else 0)
        self.files = calibration_files
        self.calibration_data = np.zeros((batch_size, self.channels, self.height, self.width), \
                                         dtype=np.float32)
        self.batch = 0
        self.preprocessor = preprocessor
        self.input_shape = input_shape

    def read_image_chw(self, path):
        from PIL import Image

        img = Image.open(path).resize((self.width, self.height), Image.NEAREST)

        im = np.array(img, dtype=np.float32, order='C')

        if len(im.shape) == 2:
            im = np.stack([im] * 3, axis=2)

        im = im[:, :, ::-1] # RGB to BGR
        im = im.transpose((2, 0, 1))

        assert len(im.shape) == 3

        return im

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch: \
                                         self.batch_size * (self.batch + 1)]
            for f in files_for_batch:
                # print("[ImageBatchStream] Processing ", f)
                img = self.read_image_chw(f)
                if self.preprocessor is not None:
                    img = self.preprocessor(img)
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])


# class PythonEntropyCalibrator( trt.IInt8EntropyCalibrator2 ) :
class PythonEntropyCalibrator( trt.IInt8MinMaxCalibrator ) :

    '''
    Reference :
        - kevin's edge_ai yolo code
        - Developer Guide 7.3
        - https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Int8/EntropyCalibrator2.html
        - https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/
        - https://forums.developer.nvidia.com/t/tensorrt-5-int8-calibration-example/71828/3
    NOTE :
        - EntropyCalibrator2 is recommended for CNN-based networks
        - reference : Developer Guide 7.3
    '''
    def __init__(self, stream, cache_file):
        # trt.IInt8EntropyCalibrator2.__init__(self)
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.stream = stream

        self.cache_file = cache_file

        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names, p_str=None):
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)

        return [self.d_input]

    def read_calibration_cache(self):
        if self.cache_file is None:
            return None

        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        if self.cache_file is not None:
            with open(self.cache_file, "wb") as f:
                f.write(cache)

class infer_helper() :
    '''
        Reference :
            - kevin's edge_ai yolo code
            - Develoepr Guide 4.3
            - /usr/src/tensorrt/samples/python/common.py
    '''

    def __init__( self, engine_path, logger, dynamic_shape=True ) :

        self.engine = load_engine( engine_path, logger )
        self.context = self.engine.create_execution_context()
        self.dynamic_shape = dynamic_shape

        if self.dynamic_shape :

            # Set max shape for cacluating output buffer size.
            self.context.set_binding_shape( 0, self.engine.get_profile_shape(0,0)[2] )
            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers( self.engine, context=self.context, dynamic_shape=True )
            self.context.set_optimization_profile_async(0, self.stream.handle )

        else :
            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers( self.engine )


        return


    def infer( self, input_batch ) :
        '''
        This function is example for when only one input/output node is needed.
        '''
        if self.dynamic_shape :
            self.context.set_binding_shape(0, input_batch.shape )

        self.inputs[0].host = input_batch.astype( np.float32, order='C' )
        [output] = common.do_inference( self.context, self.bindings, self.inputs, self.outputs, self.stream )

        return output

def load_engine( engine_path, logger ) :

    with trt.Runtime(logger) as runtime, open( engine_path, 'rb' ) as f :
        serialized_engine = f.read()
        return runtime.deserialize_cuda_engine( serialized_engine )


