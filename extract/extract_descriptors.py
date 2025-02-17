import argparse
import os
import os.path as osp
import pickle
from functools import partial

import torch
from torch.utils.data import SequentialSampler, BatchSampler

from extract_dino import extract as extract_dino, load_dinov2
from extract_cvnet import extract as extract_cvnet, load_cvnet
from image_dataset import read_imlist, DataSet, FeatureStorage
from spatial_attention_2d import SpatialAttention2d


def main():
    parser = argparse.ArgumentParser(description='Generate 1M embedding')
    parser.add_argument('--weight',
                        help='Path to weight')
    parser.add_argument('--detector', default='', help='Path to detector')

    parser.add_argument('--save_path', default='data', type=str)
    parser.add_argument('--data_path', type=str)


    parser.add_argument('--dataset',
                        help='dataset')
    parser.add_argument('--split', nargs='?', const='', default='', type=str)
    parser.add_argument('--desc_type', default='cls,global,local', type=str)
    parser.add_argument('--backbone', default='dinov2', type=str)
    parser.add_argument('--topk', default=700, type=int)
    parser.add_argument('--imsize', type=int)
    parser.add_argument('--num_workers', default=8, type=int)


    args = parser.parse_args()
    file_name = f'{args.dataset}{args.split}.txt'
    dataset, imsize, topk = args.dataset, args.imsize, args.topk
    config_file = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'conf', 'dataset', f'{dataset}.yaml')
    with open(config_file, 'w') as f:
        f.write(f'''defaults:
  - test_dataset

name: '{dataset}'

# test_gnd_file: 'gnd_{dataset}.pkl'
''')

    save_path = osp.join(args.save_path, dataset.lower())
    im_paths = read_imlist(osp.join(save_path, file_name))

    if args.backbone == 'dinov2':
        global_dim = local_dim = 768
        extract_f = partial(extract_dino, im_paths=im_paths)
        model = load_dinov2()
        scale_list = [1.]
        ps = 14
    elif args.backbone == 'cvnet':
        global_dim = 2048
        local_dim = 1024
        extract_f = extract_cvnet
        model = load_cvnet(args.weight)
        scale_list = [0.7071, 1.0, 1.4142]
        ps = None
    else:
        raise ValueError(f"Backbone {args.backbone} not supported")

    model.cuda()
    model.eval()

    detector = None
    if osp.exists(args.detector):
        detector = SpatialAttention2d(local_dim)
        detector.cuda()
        detector.eval()
        cpt = torch.load(args.detector, weights_only=False)
        detector.load_state_dict(cpt['state'], strict=True)

    data_path = osp.join(args.data_path, dataset, 'jpg')
    if args.split == '_query' and dataset in ['roxford5k', 'rparis6k', 'instre']:
        with open(osp.join(args.data_path, f'gnd_{dataset.lower()}.pkl'), 'rb') as fin:
            gnd = pickle.load(fin)['gnd']
        dataset = DataSet(dataset, data_path, scale_list, im_paths, imsize=imsize, gnd=gnd, patch_size=ps)
    else:
        dataset = DataSet(dataset, data_path, scale_list, im_paths, imsize=imsize, patch_size=ps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=BatchSampler(SequentialSampler(dataset), batch_size=1, drop_last=False),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    file_name = '' if file_name[-4:] == '.txt' else '_' + file_name

    feature_storage = FeatureStorage(save_path, args.backbone, args.split, file_name, global_dim, local_dim,
                                     len(dataset), args.desc_type, topk=topk)
    feature_storage.set_names(dataloader.dataset.data_path)
    extract_f(model, detector, feature_storage, dataloader, topk)


if __name__ == "__main__":
    main()