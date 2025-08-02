import os
from  glob import glob
import shutil
from tqdm import tqdm
import json 
from os.path import join as path_join

def load_json(path: str):
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data
    
  

def create_folder_from_img_lt(image_lt: list, output_folder: str):
    
    with tqdm(total=len(image_lt), desc='Joining folders...', unit='img') as pbar:  
        for image_path in image_lt:
            image_name = os.path.basename(image_path)
            image_name_no_ext = image_name.split('.')[0]
            new_folder_path = path_join(output_folder, image_name)
            
            shutil.copy(image_path, new_folder_path)
            pbar.update(1)
        print(f"Created folders and copied images to {output_folder}")
        
def join_coco_annotation_json(json1_path: str, json2_path: str, final_json_path: str):
    
    data1 = load_json(json1_path)
    data2 = load_json(json2_path)
    
    images1_lt = data1['images']
    anno1_lt = data1['annotations']
    cat1_lt = data1['categories']
    images2_lt = data2['images']
    anno2_lt = data2['annotations']
    
    final_dict = {
        'images': images1_lt+images2_lt,
        'annotations': anno1_lt+anno2_lt,
        'categories': cat1_lt
    }
    
    with open(final_json_path, 'w') as f:
        json.dump(final_dict, f, indent=4)
    
    print(f'Segmentation stats...\n')    
    print(f'Total json1 images: {len(data1["images"])}')
    print(f'Total json2 images: {len(data2["images"])}')
    print(f'Total combined images: {len(final_dict["images"])}\n')
    
    print(f'Total json1 annotations: {len(data1["annotations"])}')
    print(f'Total json2 annotations: {len(data2["annotations"])}')
    print(f'Total combined annotations: {len(final_dict["annotations"])}')
    print(f'Saved combined json to: {final_json_path}')    
    

if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = path_join(this_dir, 'data')
    mmdet_datasets_dir = path_join(data_dir, 'mmdet_datasets')
    endoscapes_dir = path_join(mmdet_datasets_dir, 'endoscapes')
    train_seg_dir = path_join(endoscapes_dir, 'train_seg')
    val_seg_dir = path_join(endoscapes_dir, 'val_seg')
    test_seg_dir = path_join(endoscapes_dir, 'test_seg')
    
    train_json_path = path_join(train_seg_dir, 'annotation_coco.json')
    val_json_path = path_join(val_seg_dir, 'annotation_coco.json')
    test_json_path = path_join(test_seg_dir, 'annotation_coco.json')
    output_json_path = path_join(endoscapes_dir, 'train_annotation_coco.json')
    
    train_images_lt = sorted(glob(path_join(train_seg_dir, '*.jpg')))
    val_images_lt = sorted(glob(path_join(val_seg_dir, '*.jpg')))
    test_images_lt = sorted(glob(path_join(test_seg_dir, '*.jpg')))
    
    train_val_images_lt = train_images_lt + val_images_lt + test_images_lt
    
    output_folder = path_join(endoscapes_dir, 'frames')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f'Train len: {len(train_images_lt)}')
    print(f'Val len: {len(val_images_lt)}')
    print(f'Test len: {len(test_images_lt)}')
    
    create_folder_from_img_lt(train_val_images_lt, output_folder)
    
    join_coco_annotation_json(json1_path=train_json_path,
                              json2_path=val_json_path,
                              final_json_path=output_json_path)