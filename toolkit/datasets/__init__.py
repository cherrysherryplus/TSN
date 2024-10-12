from .vot import VOTDataset
from .otb import OTBDataset
from .dtb import DTBDataset
from .uav import UAVDataset
from .uav10fps import UAV10Dataset
from .uav20l import UAV20Dataset
from .visdrone1 import VisDrone2019Dataset
from .latot import LaTOTDataset
from .latot_full import LaTOTfullDataset
from .uav123_small import uav123_smallDataset
from .uav123_big import uav123_bigDataset
from .uav123_all import uav123_allDataset
from .lasot import LaSOTDataset
from .nfs import NFSDataset
from .got10k import GOT10kDataset
from .small90 import Small90Dataset
from .small112 import Small112Dataset


class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB' in name:
            dataset = OTBDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'DTB70' in name:
            dataset = DTBDataset(**kwargs)
        elif 'UAV10fps' in name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAV20l' in name:
            dataset = UAV20Dataset(**kwargs)
        elif 'UAV' in name:
            dataset = UAVDataset(**kwargs)
        elif 'VisDrone2019' in name:
            dataset = VisDrone2019Dataset(**kwargs)
        elif 'LaTOTfull' in name:
            dataset = LaTOTfullDataset(**kwargs)
        elif 'LaTOT' in name:
            dataset = LaTOTDataset(**kwargs)
        elif 'uav123_small' in name:
            dataset = uav123_smallDataset(**kwargs)
        elif 'uav123_all' in name:
            dataset = uav123_allDataset(**kwargs)
        elif 'uav123_big' in name:
            dataset = uav123_bigDataset(**kwargs)
        elif 'nfs' in name:
            dataset = NFSDataset(**kwargs)
        elif 'VOT2018' == name or 'VOT2016' == name:
            dataset = VOTDataset(**kwargs)
        elif 'VOT2018-LT' == name:
            dataset = VOTLTDataset(**kwargs)
        elif 'small90' == name:
            dataset = Small90Dataset(**kwargs)
        elif 'small112' == name:
            dataset = Small112Dataset(**kwargs)
        elif 'TrackingNet' == name:
            dataset = TrackingNetDataset(**kwargs)
        elif 'GOT-10k' == name:
            dataset = GOT10kDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

