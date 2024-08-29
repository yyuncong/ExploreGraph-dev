import shutil
import os
import json


def merge_datasets(merge_from, merge_to):
    questions_from = os.listdir(merge_from)
    print(merge_from)
    print(len(questions_from))
    if 'log.log' in questions_from:
        print('yes')
    questions_to = os.listdir(merge_to)
    print(merge_to)
    print(len(questions_to))
    print(len(set(questions_from)&set(questions_to)))
    
    for qf in questions_from:
        frompath = os.path.join(merge_from, qf)
        if not os.path.isdir(frompath):
            continue
        topath = os.path.join(merge_to,qf)
        if os.path.exists(topath):
            print(f"Removing conflicting directory: {topath}")
            #shutil.rmtree(topath)

def checkclass_set(obj_bbox_dir, map_path):
    obj_bbox_files = os.listdir(obj_bbox_dir)
    obj_bbox_files = [f for f in obj_bbox_files if f.endswith('.json')]
    obj_class_set = set()
    for obj_bbox_file in obj_bbox_files:
        with open(os.path.join(obj_bbox_dir, obj_bbox_file), 'r') as f:
            obj_bbox = json.load(f)
        for obj in obj_bbox:
            obj_class = obj['class_name']
            obj_class_set.add(obj_class)
    with open(map_path, 'r') as f:
        class_map = json.load(f)
    class_set = set(class_map.keys())
    print(len(obj_class_set))
    print(len(class_set))
    print(len(obj_class_set&class_set))
              
        

if __name__ == "__main__":
    '''
    exploration_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/"
    merge_datasets(
        os.path.join(exploration_path,'exploration_data_2.5_best'),
        os.path.join(exploration_path,'exploration_data_2.5_best_fixed')
    )
    '''
    obj_bbox_dir = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/hm3d_obj_bbox_all"
    map_path = "bbox_mapping/matterport_category_map.json"
    checkclass_set(obj_bbox_dir, map_path)