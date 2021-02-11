from config import config
from transformers import AutoTokenizer
from Trainer import LightningModel
from pytorch_lightning.callbacks import EarlyStopping, ProgressBar, ModelCheckpoint
import pytorch_lightning as pl
import os


if __name__=="__main__":

    # Checkpoints Directory
    path = os.path.join(os.getcwd(), "checkpoints")
    
    # If there is no checkpoints folder, then create one
    if not os.path.isdir(path):
        os.mkdir(path)
        
    early_stopping = EarlyStopping(
        monitor=config["monitor"],
        min_delta=config["min_delta"],
        patience=5,
    )
    checkpoints = ModelCheckpoint(
        filepath=config["filepath"],
        monitor=config["monitor"],
        save_top_k=1
    )

    # Create the lighening model - refer the documentation. 
    model = LightningModel(config=config)
    
    
    trainer_dialog_act = pl.Trainer(
#        gpus=[],
        checkpoint_callback=checkpoints,
        callbacks=[early_stopping],
        default_root_dir="./models/",
        max_epochs=config["epochs"],
        precision=config["precision"],
    )
    
        
    trainer_dialog_act.fit(model)
    
    #trainer_dialog_act.test(model)
