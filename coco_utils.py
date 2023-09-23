from pycocotools.coco import COCO

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