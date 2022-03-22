
# Step 0. Download datasets & install packages

##  0-1. download dataset
```bash
cd ./tensorrt_example
sshpass -p guest scp -r guest@10.101.0.37:/media/root/hdd8t1/dataset/DIV2K.tar .
tar -xvf DIV2K.tar
sshpass -p guest scp -r guest@10.101.0.37:/media/root/hdd8t1/dataset/BSDS300-images.tgz .
tar -xvf BSDS300-images.tgz

```

## 0-2. download packages

```bash
pip install -r requirements.txt
```

## 0-3. download models

```bash
cd ./tensorrt_example
mkdir pytorch_model
wget https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar -P pytorch_model/
cd pytorch_model
tar -xvf model_pytorch.tar
```


# Step 1. exporting from pytorch model to onnx model

```bash
cd ./tensorrt_example
python export_onnx.py --pre_train pytorch_model/EDSR_x4.pt --input_shape 1 3 340 510 --dynamic_shape --model EDSR --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --n_threads 1 --test_only
```

# Step 2. exporting from onnx model to TRT engine
```bash
cd ./tensorrt_example
python export_trt_engine.py --dynamic_shape
```

# (Optional) Step 3. inspecting trt engine. Check which layers are not quantized.
```bash
cd ./tensorrt_example
python inspect_engine.py
```

# Step 4. Inference
```bash
cd ./tensorrt_example
python infer_engine.py --dynamic_shape
```

