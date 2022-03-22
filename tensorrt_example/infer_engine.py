import tensorrt as trt

import helper
import glob
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dynamic_shape', action='store_true' )
args = parser.parse_args()

# ----- NOTE : you can modify here ------------------------ #
engine_path = 'trt_model/EDSR_x4.engine'
DYNAMIC_SHAPE = args.dynamic_shape
test_files = glob.glob('DIV2K/DIV2K_test_LR_bicubic/X4/*.png')

def preprocessor( img ) :
    img = img
    return img

def postprocessor( img ) :
    img = img
    return img

# --------------------------------------------------------- #

def infer( test_file ) :

    i_pil_img = Image.open( test_file )   # i_pil_img.size = (W,H)

    # If i_pil_img is portrait, convert to landscape image.
    is_portrait = i_pil_img.size[0]<i_pil_img.size[1]

    if is_portrait : 
        i_pil_img = i_pil_img.transpose( Image.TRANSPOSE )

    if not DYNAMIC_SHAPE :
        i_shape = infer_helper.engine.get_binding_shape(0)  # (N,C,H,W)

        if i_pil_img.size != i_shape[2::-1] :
            i_pil_img = i_pil_img.resize( i_shape[2::-1], Image.BICUBIC )

    i_img = preprocessor( np.array( i_pil_img ) )

    # i_batch have to has shape with [N,C,H,W]
    i_batch = np.transpose( i_img, [2,0,1] )[np.newaxis]

    output = infer_helper.infer( i_batch )

    # output have to be reshapped manually
    i_shape = i_batch.shape
    if DYNAMIC_SHAPE :
        output = output[:3*i_shape[2]*4*i_shape[3]*4]   # we needs to take partial of output buffer because we set output buffer with maximum_shape of profile

    o_img = output.reshape( (3, i_shape[2]*4, i_shape[3]*4 ) )
    o_img = np.transpose( o_img, [1,2,0] )

    o_img = postprocessor( o_img )

    o_img = np.clip( o_img+0.5, 0, 255 ).astype( np.uint8 )

    if is_portrait : 
        o_img = np.transpose(o_img, [1,0,2])

    return output

def main() :

    logger = trt.Logger(trt.Logger.WARNING)

    infer_helper = helper.infer_helper( engine_path, logger, dynamic_shape=DYNAMIC_SHAPE )

    infer_helper.run( test_files[0] )   # The first batch run takes longer because of loading engine. we need to intialize before measuring timing. ( QuickStart Guide 4.5 )

    for test_file in test_files :

        o_img = infer( test_file )

        print(test_file)

    infer_helper = None

if __name__ == '__main__':
    main()

