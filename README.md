


## 环境配置说明

    conda create -n car_detection python=3.11

    conda activate car_detection

    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm

    pip3 install timm

    pip3 install albumentations

    pip install wandb scipy

*如果你是英伟达显卡，可以使用如下的命令实时监控显存的使用情况*

*watch -n 2 nvidia-smi*




