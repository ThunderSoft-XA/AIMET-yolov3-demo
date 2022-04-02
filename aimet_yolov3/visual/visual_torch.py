import copy
import torch

from torchvision import models
import urllib3

from aimet_torch.cross_layer_equalization import equalize_model

from aimet_torch import batch_norm_fold
from aimet_torch import visualize_model

from aimet_common.utils import start_bokeh_server_session
import urllib3

def visualize_changes_in_model_after_and_before_cle():
    """
    Code example for visualizating model before and after Cross Layer Equalization optimization
    """
    model = models.resnet18(pretrained=True).to(torch.device('cpu'))
    model = model.eval()
    print(model)
    # Create a copy of the model to visualize the before and after optimization changes
    model_copy = copy.deepcopy(model)

    # Specify a folder in which the plots will be saved
    results_dir = './visualization'

    batch_norm_fold.fold_all_batch_norms(model_copy, (1, 3, 224, 224))

    equalize_model(model, (1, 3, 224, 224))
    visualize_model.visualize_changes_after_optimization(model_copy, model, results_dir)

def visualize_weight_ranges_model():
    """
    Code example for model visualization
    """
    model = models.resnet18(pretrained=True).to(torch.device('cpu'))
    model = model.eval()

    # Specify a folder in which the plots will be saved
    #results_dir = './visualization'
    visualization_url, process = start_bokeh_server_session(8888)
    # call above func,then excute "bokeh serve --allow-websocket-origin=127.0.0.1:8888 --port=8888"
    # start up .py file at another animate
    # http://127.0.0.1:8888/?&bokeh-session-id=xxxx   id = (optimization/ompression)

    batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))

    # Usually it is observed that if we do BatchNorm fold the layer's weight range increases.
    # This helps in visualizing layer's weight
    visualize_model.visualize_weight_ranges(model, visualization_url)


def visualize_relative_weight_ranges_model():
    """
    Code example for model visualization
    """
    model = models.resnet18(pretrained=True).to(torch.device('cpu'))
    model = model.eval()

    # Specify a folder in which the plots will be saved
    # visualization_url, process = start_bokeh_server_session(8080)
    results_dir = '/home/thundersoft/Desktop/2022/Q1-3/aimet_yolov3/visualization'

    batch_norm_fold.fold_all_batch_norms(model, (1, 3, 224, 224))

    # Usually it is observed that if we do BatchNorm fold the layer's weight range increases.
    # This helps in finding layers which can be equalized to get better performance on hardware
    visualize_model.visualize_relative_weight_ranges_to_identify_problematic_layers(model, results_dir,display=False)

if __name__ == "__main__":
    visualize_weight_ranges_model()