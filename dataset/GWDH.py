import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

class WheatDataset(Dataset):
    """A dataset example for GWC 2021 competition."""

    def __init__(self, csv_file, root_dir, image_set, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional data augmentation to be applied
                on a sample.
        """

        annotations = pd.read_csv(csv_file)
        self.image_set = image_set
        self.image_path = root_dir+annotations["image_name"]
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        self.domains_str = annotations['domain']
        
        if(image_set == 'train'):
          self._domains = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17')
        elif(image_set == 'val'):
          self._domains = ('18', '19', '20', '21', '22', '23', '24', '25')
        else:
          self._domains = ('26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46')
        
        self.num_domains = len(self._domains)
        self.num_classes = 1
        self._domain_to_ind = dict(zip(self._domains, range(len(self._domains))))
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        
        imgp = self.image_path[idx]
        bboxes = self.boxes[idx]
        img = cv2.imread(imgp)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Opencv open images in BGR mode by default
        
        try:
          domain = torch.tensor(self._domain_to_ind[str(self.domains_str[idx])])
        except:
          domain = torch.tensor(-1)
          
        try:
          if self.transform:
              transformed = self.transform(image=image,bboxes=bboxes,class_labels=["wheat_head"]*len(bboxes)) 
              image_tr = transformed["image"]/255.0
              bboxes = transformed["bboxes"]
        except Exception as e:
          print("Execpetion: ", e, imgp)

        if len(bboxes) > 0:
          bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
          bboxes = torch.zeros((0,4))
          
               
        return image_tr, bboxes, domain, image
              
    def decodeString(self,BoxesString):
      """
      Small method to decode the BoxesString
      """
      if BoxesString == "no_box":
          return np.zeros((0,4))
      else:
          try:
              boxes =  np.array([np.array([int(i) for i in box.split(" ")])
                              for box in BoxesString.split(";")])
              return boxes.astype(np.int32).clip(min=0)
          except:
              print(BoxesString)
              print("Submission is not well formatted. empty boxes will be returned")
              return np.zeros((0,4))

