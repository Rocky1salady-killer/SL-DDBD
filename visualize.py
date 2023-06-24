import cv2
import torch
import argparse
import numpy as np
from config import get_config
from models import build_model
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def reshape_transform(tensor, height=12, width=12):

    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

if __name__ == "__main__":
    # 加载模型，参考模型推理的流程
    args, config = parse_option()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    model.to(DEVICE)

    # 处理图片
    imgPath = args.data_path
    rgb_img = cv2.imread(imgPath, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # 构建CAM
    target_layer = [model.norm]
    class_map = {0: "level_1", 1: "level_2", 2: "level_3"}
    class_id = 1
    class_name = class_map[class_id]
    cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=reshape_transform, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_id)])#[ClassifierOutputTarget(class_id)]
    grayscale_cam = grayscale_cam[0, :]

    # 可视化
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(20, 8))
    plt.subplot(121)
    plt.imshow(rgb_img)
    plt.title("origin image")

    plt.subplot(122)
    plt.imshow(visualization)
    plt.title(class_name)
    plt.show()
    plt.pause(2)
    plt.close()
