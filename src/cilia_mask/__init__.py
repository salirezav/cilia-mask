__version__ = "0.0.1"

import os, sys
from ._reader import napari_get_reader
from ._widget import OF_widget, AR_widget, threshold_magic_widget
from napari_plugin_engine import napari_hook_implementation

# , ExampleQWidget, ImageThreshold, threshold_autogenerate_widget, threshold_magic_widget
from ._writer import write_multiple, write_single_image

# from . import optflow_AR_utils


__all__ = (
    "OF_widget",
    "AR_widget",
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "threshold_magic_widget",
    #    , "ExampleQWidget", "ImageThreshold", "threshold_autogenerate_widget", "threshold_magic_widget"
)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [
        {"widget": AR_widget, "visible": True, "name": "AR Widget"},
        {"widget": OF_widget, "visible": True, "name": "OF Widget"},
    ]
