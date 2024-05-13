"""
@author: RedHotTensors
@title: ComfyUI-Prediction
@nickname: ComfyUI-Prediction
@description: Fully customizable Classifer Free Guidance for ComfyUI
"""

from .nodes import nodes_pred, nodes_sigma

NODE_CLASS_MAPPINGS = {
    **nodes_pred.NODE_CLASS_MAPPINGS,
    **nodes_sigma.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **nodes_pred.NODE_DISPLAY_NAME_MAPPINGS,
    **nodes_sigma.NODE_DISPLAY_NAME_MAPPINGS,
}
