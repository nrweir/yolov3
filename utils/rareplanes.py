import solaris as sol
import numpy as np
import os
import random
import torch

import rasterio
from albumentations.core.composition import Compose, BboxParams
from torch.utils.data import Dataset


class RarePlanesDataset(Dataset):
    """Dataset object for the RarePlanes project."""

    def __init__(self, label_dir, im_dir, starting_epoch=1,
                 batch_size=16, ims_per_epoch=3200, augs=None, dtype=None,
                 shuffle=True):
        """Create a RarePlanesDataset object.

        Arguments
        ---------
        label_dir : str
            The directory containing YOLOv3-formatted label files. For
            RarePlanes DA, the dir is usually called "tile_label_txts".
        im_dir : str
            The directory containing image files for the current stage of
            training.
        starting_epoch : int
            The epoch to start training at. If a different one is picked,
            training will start at a later set of tiles from the label set.
        batch_size : int
            Self-explanatory.
        ims_per_epoch : int
            The number of images to use in each epoch. This should not
            exceed the number of label files present in each epoch's directory.
        augs : dict
            A dictionary in the same format as specified by solaris's configs
            for augmentations. This is processed using solaris's augmentation
            processor to create an albumentations Composer object to run
            augmentations with.
        dtype : `str` or np.dtype, optional
            Allows you to specify  the dtype of your data. If not specified,
            defaults to ``np.float32``.
        shuffle : bool, optional
            Should the order of samples be shuffled within the epoch? Note
            that the tiles were generated randomly, so they're already random;
            using this flag makes training _LESS_ reproducible.

        """
        super().__init__()
        self.label_dir = label_dir
        self.im_dir = im_dir
        self.starting_epoch = starting_epoch
        self.next_epoch = starting_epoch  # incremented in self.on_epoch_end()
        self.batch_size = batch_size
        self.im_size = im_size
        self.ims_per_epoch = ims_per_epoch
        self.n_batches = ims_per_epoch//batch_size
        self.shuffle = shuffle
        if dtype is None:
            self.dtype = np.float32
        elif isinstance(dtype, str):
            try:
                self.dtype = getattr(np, dtype)
            except AttributeError:
                raise ValueError(
                    'The data type {} is not supported'.format(dtype))
        # lastly, check if it's already defined in the right format for use
        elif issubclass(dtype, np.number) or isinstance(dtype, np.dtype):
            self.dtype = dtype

        # augmentations should be passed in with a solaris-formatted
        # config dict
        aug_list = sol.nets.transform.get_augs(augs)
        self.augmenter = Compose(aug_list, bbox_params=BboxParams(
            format='yolo', label_fields=['classes']))

        self.on_epoch_end()

    def on_epoch_end(self):
        """Queue up the next set of geometries.

        ### MAKE SURE THIS GETS CALLED BETWEEN EPOCHS!!! ###
        """
        # get the epoch label file fnames from their sub-directory
        self.ep_labels = [f for f in
                          os.listdir(self.label_dir, 'ep_' + str(self.next_epoch))
                          if f.endswith('.txt')]

        # shuffle if randomly ordering; sort by sample idx if not
        if self.shuffle:
            random.shuffle(self.ep_labels)
        else:
            sample_id = [int(i.split('-')[1]) for i in self.ep_labels]
            sort_order = np.argsort(sample_id)
            self.ep_labels = [self.ep_labels[i] for i in sort_order]

        # get all the metadata that'll be needed for pulling image tiles
        self.ep_collects = [i.split('-')[2] for i in self.ep_labels]
        self.ep_x_starts = [i.split('-')[3] for i in self.ep_labels]
        self.ep_y_starts = [i.split('-')[4] for i in self.ep_labels]
        self.ep_widths = [i.split('-')[5] for i in self.ep_labels]
        self.ep_heights = [i.split('-')[6] for i in self.ep_labels]

        self.next_epoch += 1

    def __len__(self):
        return self.ims_per_epoch

    def __getitem__(self, idx):
        bboxes, classes = self.get_labels(idx)
        im = self.get_im_tile(idx)
        im_path = os.path.join(self.im_dir, self.ep_collects[idx] + '.tif')

        sample = {'image': im, 'bboxes': bboxes, 'classes': classes}
        augmented = self.augmenter(**sample)

        n_labels = len(sample['bboxes'])
        labels_out = torch.zeros((n_labels, 6))
        if n_labels:
            labels_out[:, 2:] = torch.from_numpy(
                np.array(augmented['bboxes']))
            labels_out[:, 1] = torch.from_numpy(
                np.array(augmented['classes']))
        img_out = augmented['image'].transpose(2, 0, 1)
        img_out = np.ascontiguousarray(img_out)
        # src_xy below is in shapely box order
        new_h, new_w = img_out.shape[:2]
        dw = (new_w-self.ep_widths[idx])/2
        dh = (new_h-self.ep_heights[idx])/2
        # i have no idea why this shapes thing matters, but the yolov3 Dataset
        # object makes it, so i am too
        shapes = (
            (self.ep_heights[idx], self.ep_widths[idx]),  # original shape
            (new_h/self.ep_heights[idx], new_w/self.ep_widths[idx]),  # amt chg
            (dw, dh))  # number of pixels of padding on each side

        return (torch.stack(img_out, 0), torch.cat(augmented['bboxes'], 0),
                im_path, shapes)

    def get_im_tile(self, idx, curr_row):
        im_path = os.path.join(self.im_dir, self.ep_collects[idx] + '.tif')
        with rasterio.open(im_path, 'r') as src:
            im_arr = src.read(window=rasterio.windows.Window(
                self.ep_x_starts[idx], self.ep_y_starts[idx],
                self.ep_widths[idx], self.ep_heights[idx]))

        return im_arr

    def get_labels(self, idx):
        """Read in labels from a pixel coordinate geojson."""

        label_file = os.path.join(self.label_dir,
                                  'ep_' + self.next_epoch,
                                  self.ep_labels[idx])
        with open(label_file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()],
                              dtype=np.float32)
        if labels.size > 0:
            # split bbox and label class for albumentations - re-combine later
            return labels[:, 1:], labels[:, 0]
        else:
            return np.array([]), np.array([])

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
