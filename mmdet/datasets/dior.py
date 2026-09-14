from .coco import CocoDataset
from mmdet.registry import DATASETS
from .api_wrappers import COCO

@DATASETS.register_module()
class DIORDataset(CocoDataset):
    '''
    1.数据转成COCO格式，直接继承COCO （CocoDataset)）数据类，修改一下类别即可
    
    '''
    CLASSES = ('golffield','Expressway-toll-station','vehicle','trainstation','chimney','storagetank',
               'ship','harbor','airplane','groundtrackfield','tenniscourt','dam','basketballcourt',
               'Expressway-Service-area','stadium','airport','baseballfield','bridge','windmill','overpass')
    
    METAINFO = {
        'classes':
        ('golffield','Expressway-toll-station','vehicle','trainstation','chimney','storagetank',
               'ship','harbor','airplane','groundtrackfield','tenniscourt','dam','basketballcourt',
               'Expressway-Service-area','stadium','airport','baseballfield','bridge','windmill','overpass'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True