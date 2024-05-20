import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model import CiliaModel
from data import CiliaData

# adjust
# - first arg to be "UnetPlusPlus"
# - second arg to be "densenet161"
# - "pooling" to be "avg"
# - "dropout" to be 0.1-0.5
model = CiliaModel("DeepLabV3Plus", "resnet50")
trainer = pl.Trainer(
    logger = TensorBoardLogger(save_dir = os.getcwd(), name = 'cilia_logs'),
    accelerator = "gpu",
    devices = [3,],
    strategy = "ddp_find_unused_parameters_true",
    max_epochs = 250,
    enable_model_summary = True,
    log_every_n_steps = 10,
    enable_checkpointing = True,
)

# adjust "flow_type" to be "opencv"
data = CiliaData()
data._setup()

# Run it!
trainer.fit(model, data)