"""
    训练RT-DETR
"""
import 
from .tools.YamlConfig import load_config








def main():
    CONFIG_FILE_PATH = '/home/next/桌面/RT-DETR/DETR_BiLSTM/train/config_files/rtdetrv2_r18vd_120e_voc.yml'
    # 加载各种config的配置
    yamlCfg = load_config(CONFIG_FILE_PATH)
    print(yamlCfg)



if __name__ == '__main__':
    main()



