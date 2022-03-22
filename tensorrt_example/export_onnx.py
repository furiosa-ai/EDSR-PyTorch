import torch
import os
import sys

sys.path.append('../src/')
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import torch.onnx

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():

    DYNAMIC_SHAPE = args.dynamic_shape

    _model = model.Model(args, checkpoint)
    device = torch.device( 'cpu' )
    _model.to( device )

    dummy_input = torch.ones( *(args.input_shape) )

    pre_train_name = os.path.split( args.pre_train )[-1]
    o_onnx_name = os.path.splitext( pre_train_name )[0] + '.onnx'

    o_dir = './onnx_model/'
    if not os.path.exists( o_dir ) :
        os.mkdir( o_dir )

    o_onnx_path = os.path.join( o_dir, o_onnx_name )

    if DYNAMIC_SHAPE :
        torch.onnx.export( _model, dummy_input, o_onnx_path, verbose=True,
                input_names=['input.0'], output_names=['output.0'],
                dynamic_axes={'input.0':[2,3]}, opset_version=12 )
    else :
        torch.onnx.export( _model, dummy_input, o_onnx_path, verbose=True,
                input_names=['input.0'], output_names=['output.0'], opset_version=12 )
    print( f'[INFO] Export succeed. : { o_onnx_path }' )

    checkpoint.done()

if __name__ == '__main__':
    main()
