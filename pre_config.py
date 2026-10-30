"""Arguments shared by the LLCM and VCM preprocessing scripts."""
import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('--datasets_path',
                        default=os.environ.get('PTC_DATA_ROOT', 'datasets'))
    parser.add_argument('--img_w', type=int, default=128)
    parser.add_argument('--img_h', type=int, default=256)
    return parser.parse_args()
