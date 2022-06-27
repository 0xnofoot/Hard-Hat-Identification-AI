from __future__ import division, print_function
import argparse

from utils.misc_utils import parse_anchors, read_class_names
from utils.plot_utils import get_color_table
parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
parser.add_argument("--input_video", type=str, default="vedio1.mp4",
                    help="The path of the input video.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[608, 608],#416 416
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str,
                    default="./checkpoint/best_model_Epoch_120_step_370864_mAP_0.9608_loss_6.4684_lr_8.803196e-05",
                    help="The path of the weights to restore.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to save the video detection results.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)