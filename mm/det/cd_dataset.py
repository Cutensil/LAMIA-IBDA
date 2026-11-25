import os.path as osp
from mm.det.datasets.coco import CocoDataset
from mm.det.registry import DATASETS

@DATASETS.register_module()
class BiCocoDataset(CocoDataset):
    def __init__(self, *args, pre_dir='pre_dir', post_dir='post_dir', **kwargs):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = super().parse_data_info(raw_data_info)
        filename = osp.basename(data_info['img_path'])
        data_info['pre_img_path'] = osp.join(self.pre_dir, filename)
        data_info['post_img_path'] = osp.join(self.post_dir, filename)
        
        return data_info
