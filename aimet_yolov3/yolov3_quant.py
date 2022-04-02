#! /usr/bin/env python3

from __future__ import division

import argparse
import tqdm
import numpy as np

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from yolov3.models import load_model
from utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info
from utils.datasets import ListDataset
from utils.transforms import DEFAULT_TRANSFORMS
from utils.parse_config import parse_data_config

from aimet_torch import quantsim

#CLE
from aimet_torch.cross_layer_equalization import equalize_model

# BC
from aimet_torch.quantsim import QuantParams
from aimet_torch.bias_correction import correct_bias

#BNF
from aimet_torch import batch_norm_fold
from aimet_common.defs import QuantScheme

from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.quantsim import QuantizationSimModel

#bokeh visualize
from aimet_common.utils import start_bokeh_server_session
from aimet_torch import visualize_model


def parse_args():
    parser = argparse.ArgumentParser(prog='pose_estimation_quanteval',
                                     description='Evaluate the post quantized SRGAN model')

    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="models/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    
    parser.add_argument('--representative-datapath',
                        '-reprdata',
                        help='The location where representative data are stored. '
                             'The data will be used for computation of encodings',
                        type=str)
    parser.add_argument('--quant-scheme',
                        '-qs',
                        help='Support two schemes for quantization: [`tf` or `tf_enhanced`],'
                             '`tf_enhanced` is used by default',
                        default='tf_enhanced',
                        choices=['tf', 'tf_enhanced'],
                        type=str)

    return parser.parse_args()

def _evaluate(model):
     """Evaluate model on validation dataset.

     :param model: Model to evaluate
     :type model: models.Darknet
     :param dataloader: Dataloader provides the batches of images with targets
     :type dataloader: DataLoader
     :param class_names: List of class names
     :type class_names: [str]
     :param img_size: Size of each image dimension for yolo
     :type img_size: int
     :param iou_thres: IOU threshold required to qualify as detected
     :type iou_thres: float
     :param conf_thres: Object confidence threshold
     :type conf_thres: float
     :param nms_thres: IOU threshold for non-maximum suppression
     :type nms_thres: float
     :param verbose: If True, prints stats of model
     :type verbose: bool
     :return: Returns precision, recall, AP, f1, ap_class
     """
     #dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose
     def func_wrapper(model,arguments):
          model.eval()  # Set model to evaluation mode

          Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

          labels = []
          sample_metrics = []  # List of tuples (TP, confs, pred)
          for imgs, targets in tqdm.tqdm(arguments[0], desc="Validating"):
               # Extract labels
               labels += targets[:, 1].tolist()
               # Rescale target
               targets[:, 2:] = xywh2xyxy(targets[:, 2:])
               targets[:, 2:] *= arguments[2]

               imgs = Variable(imgs.type(Tensor), requires_grad=False)

               with torch.no_grad():
                    outputs = model(imgs)
                    outputs = non_max_suppression(outputs, conf_thres=arguments[4], iou_thres=arguments[5])

               sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=arguments[3])

          if len(sample_metrics) == 0:  # No detections over whole validation set.
               print("---- No detections over whole validation set ----")
               return None

          # Concatenate sample statistics
          true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
          metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)

          print_eval_stats(metrics_output, arguments[1], arguments[6])
          return metrics_output
     return func_wrapper

def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- precision {precision.mean():.5f} ----")
        print(f"---- recall {recall.mean():.5f} ----")
        print(f"---- mAP {AP.mean():.5f} ----")
        print(f"---- f1 {f1.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")

def yolov3_quant_eval(args):
     # Load configuration from data file
     data_config = parse_data_config(args.data)
     # Path to file containing all images for validation
     valid_path = data_config["valid"]
     class_names = load_classes(data_config["names"])
     dataset = ListDataset(valid_path, img_size=args.img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
     dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)

     model = load_model(args.model, args.weights)

     eval_func_quant = _evaluate(model)
     eval_func = _evaluate(model)

     sim = quantsim.QuantizationSimModel(model, dummy_input=torch.Tensor(1, 3, 416, 416))

     #sim.compute_encodings(forward_pass_callback=eval_func_quant, 
     #     forward_pass_callback_args=(dataloader,class_names,args.img_size,args.iou_thres,args.conf_thres,args.nms_thres,True))

     visualization_url, process = start_bokeh_server_session(8888)
     # call above func,then excute "bokeh serve --allow-websocket-origin=127.0.0.1:8888 --port=8888"
     # start up .py file at another animate
     # http://127.0.0.1:8888/?&bokeh-session-id=xxxx   id = (optimization/ompression)

     batch_norm_fold.fold_all_batch_norms(model, (1, 3, 416, 416))

     # Usually it is observed that if we do BatchNorm fold the layer's weight range increases.
     # This helps in visualizing layer's weight
     visualize_model.visualize_weight_ranges(model, visualization_url)

     # return precision, recall, AP, f1, ap_class
     print(f'The [precision, recall, AP, f1, ap_class] results are: eval_num')

from aimet_torch.onnx_utils import OnnxSaver, OnnxExportApiArgs
def yolov3_cle_bc(args):
     # Load configuration from data file
     data_config = parse_data_config(args.data)
     # Path to file containing all images for validation
     valid_path = data_config["valid"]
     class_names = load_classes(data_config["names"])
     dataset = ListDataset(valid_path, img_size=args.img_size, 
          multiscale=False, transform=DEFAULT_TRANSFORMS)
     dataloader = DataLoader(
          dataset,
          batch_size=args.batch_size,
          shuffle=False,
          num_workers=args.n_cpu,
          pin_memory=True,
          collate_fn=dataset.collate_fn)

     model = load_model(args.model, args.weights)

     _ = batch_norm_fold.fold_all_batch_norms(model, (1, 3, 416, 416))
     use_cuda = False
     dummy_input = torch.rand(1, 3, 416, 416)
     if use_cuda:
          dummy_input = dummy_input.cuda()

     sim = QuantizationSimModel(model=model,
                              quant_scheme=QuantScheme.post_training_tf_enhanced,
                              dummy_input=dummy_input,
                              default_output_bw=8,
                              default_param_bw=8)

     print(sim.model)
     print(sim)

     eval_func_quant = _evaluate(model)

     sim = quantsim.QuantizationSimModel(model, dummy_input=torch.Tensor(1, 3, 416, 416))

     sim.compute_encodings(forward_pass_callback=eval_func_quant, 
          forward_pass_callback_args=(dataloader,class_names,
               args.img_size,args.iou_thres,args.conf_thres,args.nms_thres,True))

     model_cle = load_model(args.model, args.weights)
     equalize_model(model_cle,input_shapes=(1,3,416,416))

     clesim = quantsim.QuantizationSimModel(model_cle, dummy_input=torch.Tensor(1, 3, 416, 416))

     clesim.compute_encodings(forward_pass_callback=eval_func_quant, 
          forward_pass_callback_args=(dataloader,class_names,args.img_size,
               args.iou_thres,args.conf_thres,args.nms_thres,True))

     bc_params = QuantParams(weight_bw=8, act_bw=8, round_mode="nearest",
          quant_scheme=QuantScheme.post_training_tf_enhanced)

     correct_bias(model_cle, bc_params, num_quant_samples=16,
          data_loader= dataloader, num_bias_correct_samples=16)

     sim = quantsim.QuantizationSimModel(model_cle, dummy_input=torch.Tensor(1, 3, 416, 416))

     sim.compute_encodings(forward_pass_callback=eval_func_quant, 
          forward_pass_callback_args=(dataloader,class_names,args.img_size,
               args.iou_thres,args.conf_thres,args.nms_thres,True))
     dummy_input = dummy_input.cpu()
     onnxparams = OnnxExportApiArgs(opset_version=11,input_names=[" input"],output_names=["106"])
     sim.export(path='./output/', filename_prefix='yolov3_after_cle_bc', dummy_input=dummy_input,onnx_export_args=onnxparams)

import os
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
def yolov3_Adaround(args):
          # Load configuration from data file
     data_config = parse_data_config(args.data)
     # Path to file containing all images for validation
     valid_path = data_config["valid"]
     class_names = load_classes(data_config["names"])
     dataset = ListDataset(valid_path, img_size=args.img_size, 
          multiscale=False, transform=DEFAULT_TRANSFORMS)
     dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)

     model = load_model(args.model, args.weights)

     eval_func_quant = _evaluate(model)
     eval_func = _evaluate(model)

     batch_norm_fold.fold_all_batch_norms(model, (1, 3, 416, 416))
     dummy_input = torch.rand(1, 3, 416, 416).cuda() if torch.has_cuda else torch.rand(1, 3, 416, 416)

     sim = quantsim.QuantizationSimModel(model,
          quant_scheme=QuantScheme.post_training_tf_enhanced,
          dummy_input=dummy_input,
          default_output_bw=8,
          default_param_bw=8)

     #sim.compute_encodings(forward_pass_callback=eval_func_quant, 
     #     forward_pass_callback_args=(dataloader,class_names,args.img_size,args.iou_thres,args.conf_thres,args.nms_thres,True))

     visualization_url, process = start_bokeh_server_session(8888)
     # call above func,then excute "bokeh serve --allow-websocket-origin=127.0.0.1:8888 --port=8888"
     # start up .py file at another animate
     # http://127.0.0.1:8888/?&bokeh-session-id=xxxx   id = (optimization/ompression)

     # Usually it is observed that if we do BatchNorm fold the layer's weight range increases.
     # This helps in visualizing layer's weight
     # visualize_model.visualize_weight_ranges(model, visualization_url)

     sim.compute_encodings(forward_pass_callback=eval_func_quant, 
          forward_pass_callback_args=(dataloader,class_names,args.img_size,
               args.iou_thres,args.conf_thres,args.nms_thres,True))

     params = AdaroundParameters(data_loader=dataloader, num_batches= 1,
          default_num_iterations=32)
     os.makedirs("./output/", exist_ok=True)
     ada_model = Adaround.apply_adaround(model,dummy_input,params,
          path="output", filename_prefix="adaround",
          default_param_bw=8,
          default_quant_scheme=QuantScheme.post_training_tf_enhanced)
     
     adsim = QuantizationSimModel(model=ada_model,
                           dummy_input=dummy_input,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           default_output_bw=8, 
                           default_param_bw=8)

     adsim.set_and_freeze_param_encodings(encoding_path=os.path.join("output", 'adaround.encodings'))
     adsim.compute_encodings(forward_pass_callback=eval_func_quant, 
          forward_pass_callback_args=(dataloader,class_names,args.img_size,
               args.iou_thres,args.conf_thres,args.nms_thres,True))

     dummy_input = dummy_input.cpu()
     onnxparams = OnnxExportApiArgs(opset_version=11) #,input_names=[" input"],output_names=["106"]
     adsim.export(path='./output/', filename_prefix='yolov3_after_adaround', dummy_input=dummy_input, onnx_export_args=onnxparams)

if __name__ == "__main__":
     args = parse_args()
#     yolov3_quant_eval(args)
     yolov3_Adaround(args)