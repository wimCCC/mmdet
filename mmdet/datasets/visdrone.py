from .coco import CocoDataset
from mmdet.registry import DATASETS
from .api_wrappers import COCO

@DATASETS.register_module()
class VDDataset(CocoDataset):
    '''
    1.数据转成COCO格式，直接继承COCO （CocoDataset)）数据类，修改一下类别即可
    
    '''
    CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
    
    METAINFO = {
        'classes':
        ('pedestrian', 'people', 'bicycle', 'car', 'van', 
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True