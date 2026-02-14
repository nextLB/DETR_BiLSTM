


import os
import yaml



def load_config(filePath, cfg=dict()):
    _, ext = os.path.splitext(filePath)
    assert ext in ['.yml', '.yaml'], "only support yaml files"

    with open(filePath) as f:
        fileCfg = yaml.load(f, Loader=yaml.Loader)
        if fileCfg is None:
            return {}

    print(fileCfg)




def main():
    TEST_CONFIG_FILE_PATH = '/home/next/桌面/RT-DETR/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_voc.yml'

    load_config(TEST_CONFIG_FILE_PATH)




if __name__ == '__main__':
    main()





