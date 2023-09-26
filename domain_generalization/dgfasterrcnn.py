import sys
sys.path.append('..')

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pytorch_lightning.core.lightning import LightningModule
from .classifiers import InstanceDomainClassifier, ImageDomainClassifier, InstanceClassifierPrime, InstanceClassifier
from dataset import WheatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fasterrcnn import fasterrcnn_resnet50_fpn
from .utils import collate_fn, convert_wheat_dataset_to_coco, accuracy
from .coco_eval import CocoEvaluator

class DGFasterRCNN(LightningModule):
    def __init__(
          self, 
          dataset_root, 
          batch_size=2,
          base_lr=1e-5,
          weight_decay=0.0001,
          momentum=0.9
    ):
        super(DGFasterRCNN, self).__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.prepare_transforms()
        self.prepapre_datasets()

        self.num_trdomains, self.num_classes = self.train_dataset.num_domains, self.train_dataset.num_classes
        self.num_valdomains = self.val_dataset.num_domains
        self.num_testdomains = self.test_dataset.num_domains

        self.detector = fasterrcnn_resnet50_fpn(min_size=1024, max_size=1024, pretrained_backbone=True)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes+1)

        self.ImageDG = ImageDomainClassifier(256, self.num_trdomains)
        self.InsDG = InstanceDomainClassifier(self.num_trdomains)       
        self.InsCls = nn.ModuleList([InstanceClassifier(self.num_classes+1) for i in range(self.num_trdomains)])
        self.InsClsPrime = nn.ModuleList([InstanceClassifierPrime(self.num_classes+1) for i in range(self.num_trdomains)])

        self.val_acc_stack = [[] for i in range(self.num_valdomains)]
        self.test_acc_stack = [[] for i in range(self.num_testdomains)]

        self.detector.backbone.register_forward_hook(self.store_backbone_out)
        self.detector.roi_heads.box_head.register_forward_hook(self.store_ins_features)

        self.mode = 0
        self.sub_mode = 0


    def prepare_transforms(self):
        
        self.train_transform = A.Compose([
            ToTensorV2(p=1.0),
        ],p=1.0,bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))
        
        self.val_transform = A.Compose([
            ToTensorV2(p=1.0),
        ],p=1.0,bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))

    def prepapre_datasets(self):
        images_path = os.path.join(self.dataset_root, 'images/')
        train_gt_path = os.path.join(self.dataset_root, 'annots', 'official_train.csv')
        val_gt_path = os.path.join(self.dataset_root, 'annots', 'official_val.csv')
        test_gt_path = os.path.join(self.dataset_root, 'annots', 'official_test.csv')

        self.train_dataset = WheatDataset(train_gt_path, root_dir=images_path, image_set='train', transform=self.train_transform)
        self.val_dataset = WheatDataset(val_gt_path, root_dir=images_path, image_set='val', transform=self.val_transform)
        self.test_dataset = WheatDataset(test_gt_path, root_dir=images_path, image_set='test', transform=self.val_transform)


    def forward(self, imgs,targets=None):
      # Torchvision FasterRCNN returns the loss during training 
      # and the boxes during eval
      self.detector.eval()
      return self.detector(imgs)

    def store_ins_features(self, module, input1, output):
      self.box_features = output
      self.box_labels = input1[1] #Torch tensor of size 512
      
            
    def store_backbone_out(self, module, input1, output):
      self.base_feat = output

    def configure_optimizers(self):
      
      optimizer = torch.optim.Adam([{'params': self.detector.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.ImageDG.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsDG.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsCls.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsClsPrime.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay}
                                      ],) 
      
      lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5, threshold=0.0001, min_lr=0, eps=1e-08),
                      'monitor': 'map@50'}
      
      
      return [optimizer], [lr_scheduler]
    
    def train_dataloader(self):
      num_train_sample_batches = len(self.train_dataset)//self.batch_size 
      temp_indices = np.array([i for i in range(len(self.train_dataset))])
      np.random.shuffle(temp_indices)

      sample_indices = []
      for i in range(num_train_sample_batches):
        batch = temp_indices[self.batch_size*i:self.batch_size*(i+1)]

        for index in batch:
          sample_indices.append(index)  
  
        for index in batch:		 
          sample_indices.append(index)

      return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sample_indices, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn, num_workers=4)
    

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False,  collate_fn=collate_fn, num_workers=4)
    
    def training_step(self, batch, batch_idx):
      imgs = list(image.cuda() for image in batch[0]) 

      targets = []
      for boxes, domain in zip(batch[1], batch[2]):
        target= {}
        target["boxes"] = boxes.float().cuda()
        target["labels"] = torch.ones(len(target["boxes"])).long().cuda()
        targets.append(target)
      
      if(self.mode == 0):
        temp_loss = []
        for index in range(len(imgs)):
          detections = self.detector([imgs[index]], [targets[index]])
          temp_loss.append(sum(loss1 for loss1 in detections[0]['losses'].values()))
     
        if(self.sub_mode == 0):
          self.mode = 1
          self.sub_mode = 1
        elif(self.sub_mode == 1):
          self.mode = 2
          self.sub_mode = 2
        elif(self.sub_mode == 2):
          self.mode = 3
          self.sub_mode = 3
        elif(self.sub_mode == 3):
          self.mode = 4
          self.sub_mode = 4  
        else:
          self.sub_mode = 0
          self.mode = 0

        loss = torch.mean(torch.stack(temp_loss))

      elif(self.mode == 1):
        
        loss_dict = {}
        temp_loss = []
        for index in range(len(imgs)):
          _ = self.detector([imgs[index]], [targets[index]])
            
          ImgDA_scores = self.ImageDG(self.base_feat['0'])
          loss_dict['DA_img_loss'] = 0.5*F.cross_entropy(ImgDA_scores, torch.unsqueeze(batch[2][index], 0))
          IDA_out = self.InsDG(self.box_features)
          loss_dict['DA_ins_loss'] = F.cross_entropy(IDA_out, batch[2][index].repeat(IDA_out.shape[0]).long())
          loss_dict['Cst_loss'] = F.mse_loss(IDA_out, ImgDA_scores[0].repeat(IDA_out.shape[0],1))
          
          temp_loss.append(sum(loss1 for loss1 in loss_dict.values()))

               
        loss = torch.mean(torch.stack(temp_loss))
        self.mode = 0

      elif(self.mode == 2): #Without recording the gradients for detector, we need to update the weights for classifier weights
        loss_dict = {}
        loss = []

        
        for index in range(len(self.InsCls)):
          for param in self.InsCls[index].parameters(): param.requires_grad = True

        for index in range(len(imgs)):
          with torch.no_grad():
            _ = self.detector([imgs[index]], [targets[index]])
          
          cls_scores = self.InsCls[batch[2][index].item()](self.box_features)
          loss.append(F.cross_entropy(cls_scores, self.box_labels[0])) 

        loss_dict['cls'] = 0.05*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 0

      elif(self.mode == 3): #Only the GRL Classification should influence the updates but here we need to update the detector weights as well
        loss_dict = {}
        loss = []
    
        for index in range(len(imgs)):
          _ = self.detector([imgs[index]], [targets[index]])
          cls_scores = self.InsClsPrime[batch[2][index].item()](self.box_features)
          loss.append(F.cross_entropy(cls_scores, self.box_labels[0]))
  	  
        loss_dict['cls_prime'] = 0.0001*(torch.mean(torch.stack(loss)))
        loss = sum(loss for loss in loss_dict.values())

        self.mode = 0
        
      else: #For Mode 4
      
        loss_dict = {}
        loss = []
        consis_loss = []
        
        for index in range(len(self.InsCls)):
          for param in self.InsCls[index].parameters(): param.requires_grad = False
        
        for index in range(len(imgs)):
          _ = self.detector([imgs[index]], [targets[index]])
          temp = []
          for i in range(len(self.InsCls)):
            if(i != batch[2][index].item()):
              cls_scores = self.InsCls[i](self.box_features)
              temp.append(cls_scores)
              loss.append(F.cross_entropy(cls_scores, self.box_labels[0]))
          consis_loss.append(torch.mean(torch.abs(torch.stack(temp, dim=0) - torch.mean(torch.stack(temp, dim=0), dim=0))))

        loss_dict['cls'] = 0.05*(torch.mean(torch.stack(loss)))# + torch.mean(torch.stack(consis_loss)))
        loss = sum(loss for loss in loss_dict.values())
        
        self.mode = 0
        self.sub_mode = 0

      return {"loss": loss}#, "log": torch.stack(temp_loss).detach().cpu()}



    def on_validation_epoch_start(self):
      coco = convert_wheat_dataset_to_coco(self.val_dataset)
      self.coco_evaluator_val = CocoEvaluator(coco, iou_types=["bbox"])

    def validation_step(self, batch, batch_idx):
      img, boxes, domain, _ = batch
      
      preds = self.forward(img)

      res = {(batch_idx+1): pred for pred in preds}

      self.coco_evaluator_val.update(res)

      preds[0]['boxes'] = preds[0]['boxes'][preds[0]['scores'] > 0.5]
      self.val_acc_stack[domain[0]].append(torch.stack([accuracy(b,pb["boxes"],iou_threshold=0.5) for b,pb in zip(boxes,preds)]))

    def on_validation_epoch_end(self):

      temp = 0
      non_zero_domains = 0
      
      for item in range(len(self.val_acc_stack)):
        
        if(self.val_acc_stack[item]):
          temp = temp + torch.mean(torch.stack(self.val_acc_stack[item]))
          non_zero_domains = non_zero_domains + 1
          print(torch.mean(torch.stack(self.val_acc_stack[item])))
          
      temp = temp/non_zero_domains #8 Validation domains 

      self.val_acc_stack = [[] for i in range(self.num_valdomains)]

      self.coco_evaluator_val.synchronize_between_processes()
      self.coco_evaluator_val.accumulate()
      self.coco_evaluator_val.summarize()

      print("Validation ADA: ", temp)

      val_map50 = self.coco_evaluator_val.coco_eval['bbox'].stats[1]
      self.log('map@50', val_map50)

    def on_test_start(self):
      coco = convert_wheat_dataset_to_coco(self.test_dataset)
      self.coco_evaluator_test = CocoEvaluator(coco, iou_types=["bbox"])
    
    def test_step(self, batch, batch_idx):
      img, boxes, domain, _ = batch
      
      preds = self.forward(img)

      res = {(batch_idx+1): pred for pred in preds}

      self.coco_evaluator_test.update(res)

      preds[0]['boxes'] = preds[0]['boxes'][preds[0]['scores'] > 0.5]
      self.test_acc_stack[domain[0]].append(torch.stack([accuracy(b,pb["boxes"],iou_threshold=0.5) for b,pb in zip(boxes,preds)]))

    def on_test_end(self):
      temp = 0
      non_zero_domains = 0
      
      for item in range(len(self.test_acc_stack)):
        
        if(self.test_acc_stack[item]):
          temp = temp + torch.mean(torch.stack(self.test_acc_stack[item]))
          non_zero_domains = non_zero_domains + 1
          print(torch.mean(torch.stack(self.test_acc_stack[item])))
      temp = temp/non_zero_domains
          
      self.test_acc_stack = [[] for i in range(self.num_testdomains)]

      self.coco_evaluator_test.synchronize_between_processes()
      self.coco_evaluator_test.accumulate()
      self.coco_evaluator_test.summarize()

      print("Test ADA: ", temp)
        