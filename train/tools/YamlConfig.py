


import os
import yaml
import copy
from typing import Any, Dict, Optional, List

INCLUDE_KEY = '__include__'




def load_config(filePath, cfg=dict()):
    _, ext = os.path.splitext(filePath)
    assert ext in ['.yml', '.yaml'], "only support yaml files"

    with open(filePath) as f:
        fileCfg = yaml.load(f, Loader=yaml.Loader)
        if fileCfg is None:
            return {}

    if INCLUDE_KEY in fileCfg:
        baseYamls = list(fileCfg[INCLUDE_KEY])
        for baseYaml in baseYamls:
            if baseYaml.startswith('~'):
                baseYaml = os.path.expanduser(baseYaml)

            if not baseYaml.startswith('/'):
                baseYaml = os.path.join(os.path.dirname(filePath), baseYaml)

            with open(baseYaml) as f:
                baseCfg = yaml.load(f, Loader=yaml.Loader)
                merge_dict(cfg, baseCfg)

    return merge_dict(cfg, fileCfg)


def merge_dict(dct, another_dct, inplace=True) -> Dict:
    def _merge(dct, another) -> Dict:
        for k in another:
            if (k in dct and isinstance(dct[k], dict) and isinstance(another[k], dict)):
                _merge(dct[k], another[k])
            else:
                dct[k] = another[k]

        return dct

    if not inplace:
        dct = copy.deepcopy(dct)
    return _merge(dct, another_dct)





def main():
    TEST_CONFIG_FILE_PATH = '/home/next/桌面/RT-DETR/DETR_BiLSTM/train/config_files/rtdetrv2_r18vd_120e_voc.yml'

    cfg = load_config(TEST_CONFIG_FILE_PATH)
    print(cfg['task'])



if __name__ == '__main__':
    main()





