import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from torchvision.models.detection._utils import Matcher
from torchvision.ops.boxes import box_iou

def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

    """

    images = list()
    targets=list()
    orig_img = list()
    domain_labels = list()
    for i, t, d, io in batch:
        images.append(i)
        targets.append(t)
        orig_img.append(io)
        domain_labels.append(d)
    images = torch.stack(images, dim=0)

    return images, targets, domain_labels, orig_img

def convert_wheat_dataset_to_coco(ds):
   coco_ds = COCO()
   dataset = {"images": [], "categories": [], "annotations": []}
   dataset["categories"] = [{"id": 1, "name": "wheat_head", "supercategory": "wheat_head"}]
   ann_id = 1
   for img_idx in range(len(ds)):
      _, bboxes, domain, image = ds[img_idx]
      img_dict = {}
      image_id = img_idx+1
      img_dict["id"] = image_id
      img_dict["width"] = image.shape[0]
      img_dict["height"] = image.shape[1]
      img_dict["domain"] = domain
      dataset["images"].append(img_dict)
      bboxes[:, 2:] -= bboxes[:, :2]
      bboxes = bboxes.tolist()
      num_objs = len(bboxes)
      for i in range(num_objs):
         ann = {}
         ann["image_id"] = image_id
         ann["bbox"] = bboxes[i]
         ann["category_id"] = 1
         ann["id"] = ann_id
         ann["iscrowd"] = 0
         ann["area"] = bboxes[i][2]*bboxes[i][3]
         dataset["annotations"].append(ann)
         ann_id += 1

   coco_ds.dataset = dataset
   coco_ds.createIndex()
   return coco_ds

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

def accuracy(src_boxes,pred_boxes ,  iou_threshold = 1.):

    total_gt = len(src_boxes)
    total_pred = len(pred_boxes)
    if total_gt > 0 and total_pred > 0:


        # Define the matcher and distance matrix based on iou
        matcher = Matcher(iou_threshold,iou_threshold,allow_low_quality_matches=False) 
        match_quality_matrix = box_iou(src_boxes,pred_boxes)

        results = matcher(match_quality_matrix)
        
        true_positive = torch.count_nonzero(results.unique() != -1)
        matched_elements = results[results > -1]
        
        #in Matcher, a pred element can be matched only twice 
        false_positive = torch.count_nonzero(results == -1) + ( len(matched_elements) - len(matched_elements.unique()))
        false_negative = total_gt - true_positive

            
        return  true_positive / (true_positive + false_positive + false_negative) 

    elif total_gt == 0:
        if total_pred > 0:
            return torch.tensor(0.).cuda()
        else:
            return torch.tensor(1.).cuda()
    elif total_gt > 0 and total_pred == 0:
        return torch.tensor(0.).cuda()