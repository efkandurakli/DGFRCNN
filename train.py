from domain_generalization import DGFasterRCNN
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


NET_FOLDER = 'GWHD'
DATASET_ROOT = '/home/duraklefkan/workspace/Datasets/gwhd_2021'
weights_file = 'best_prop'

seed_everything(25081992)

early_stop_callback= EarlyStopping(monitor='map@50', min_delta=0.00, patience=10, verbose=False, mode='max')
checkpoint_callback = ModelCheckpoint(monitor='map@50', dirpath=NET_FOLDER, filename=weights_file)

detector = DGFasterRCNN('/home/duraklefkan/workspace/Datasets/gwhd_2021')

trainer = Trainer(accelerator='gpu', devices=1, enable_progress_bar=True, max_epochs=100, 
                  deterministic=False, reload_dataloaders_every_n_epochs=1, 
                  callbacks=[checkpoint_callback, early_stop_callback],
                  num_sanity_val_steps=0)

trainer.fit(detector)