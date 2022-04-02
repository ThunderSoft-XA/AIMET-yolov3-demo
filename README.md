## AIMET Setup Developer documentation

The main purpose of this project is to guide users to quickly master AIMET APIs, so that users can optimize their models more smoothly.

Due to the problem of my network and host resources, I installed the 1.16.2.py37 version of torch_cpu on the host using conda. This is not the official recommended installation method, so I introduced this installation method in detail in the document, and I think this method should be universally applicable. 

## The main development process:
* Build aimet environment in conda 
* Prepare the data set and torch yolov3
* Quantizate yolov3 structure

## Build aimet environment in conda
AIMET github project :
https://github.com/quic/aimet

I have provided a Yaml (aimet_env.yaml)file for the Conda environment in the project root directory that you can use to create a new Conda environment.
`conda env create -f aimet_env.yaml`

Import the following environment variables into your userspace ~ /.bashrc file:
```bash
# AIMET CONFIG

export PATH="/home/xxx/xxx/anaconda3/envs/aimet/lib/python3.7/site-packages/aimet_common/x86_64-linux-gnu:$PATH"
# export PATH="/home/xxx/xxx/anaconda3/envs/aimet/lib/python3.7/site-packages/aimet_common/x86_64-linux-gnu/aimet_tensor_quantizer-0.0.0-py3.7-linux-x86_64.egg:$PATH"
export LD_LIBRARY_PATH="/home/xxx/xxx/anaconda3/envs/aimet/lib/python3.7/site-packages/aimet_common/x86_64-linux-gnu:$LD_LIBRARY_PATH"
if [[ $PYTHONPATH = "" ]]; then 
    export PYTHONPATH="/home/xxx/xxx/anaconda3/envs/aimet/lib/python3.7/site-packages/aimet_common/x86_64-linux-gnu;"
else 
    export PYTHONPATH="/home/xxx/xxx/anaconda3/envs/aimet/lib/python3.7/site-packages/aimet_common/x86_64-linux-gnu:$PYTHONPATH"
fi
```

Download the torch-cpu-1.16.2.py37 package corresponding to aimet :

https://github.com/quic/aimet/releases

## Prepare the data set and torch yolov3
The Torch version of Yolov3 comes from the following open source projects:

https://github.com/eriklindernoren/PyTorch-YOLOv3

Follow the steps in the open source project to get the data set and network build scripts.

copy yolov3.weights model file to aimet_yolov3/models
```
The model file is too large,I did not upload it .
```

Copy the pictures and corresponding labels files to the aimet_yolov3/data/images and aimet_yolov3/data/label folders respectively 

## quantizate yolov3 structure

yolov3 network was selected and its PyTorch version model was quantified. Darknet53 of Yolov3 includes a large number of DBL structures (consisting of Convolution + BatchNormalization + LeakyReLU nodes) and hopefully this structure will have some good results under AIMET's BNF operation. This is the main reason why yolov3 was chosen for quantification in  the AIMET demo.

COCO data set was used, but not all data were used in the quantization process, only 5K, 0.6K, 0.2K, 0.1K and 40 were used (When the size of data set was 5K, all parameters of quantization aware training were slightly reduced compared with the original model. Meanwhile, due to insufficient hardware resources, in order to reduce the quantization time,greatly reduce the amount of data) The evaluation function returns four main model evaluation parameters: Precision, recall, mAP and F1 Score (a harmonic mean of model accuracy and recall). When the data set was 40, I obtained a good F1 Score value, so the size of the data set for subsequent operations was selected as 40. 

The effects of BNF operation alone on Yolov3 are slightly less than those of CLE operation alone, but both of them are better than the evaluation parameters of the original model. Therefore, two combined operations of BNF--CLE--BC--QAT and BNF--AdaRound--QAT are adopted for quantization operation, both of which bring improvement. But obviously BNF--CLE--BC--QAT quantification results are more satisfactory.

Can execute python scripts directly:
```
conda activate aimet_env
python yolov3_quant.py
````

You can see all the evaluation results at res/result.