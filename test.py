import torch    
from domain_generalization import DGFasterRCNN
from pytorch_lightning import Trainer


NET_FOLDER = 'GWHD'
DATASET_ROOT = '/home/duraklefkan/workspace/Datasets/gwhd_2021'
weights_file = 'best_prop-v3'


detector = DGFasterRCNN('/home/duraklefkan/workspace/Datasets/gwhd_2021')
detector.load_state_dict(torch.load(NET_FOLDER+'/'+weights_file+'.ckpt')['state_dict'])
detector.freeze()

trainer = Trainer(accelerator='gpu', devices=1, enable_progress_bar=True, num_sanity_val_steps=0)

trainer.test(detector)