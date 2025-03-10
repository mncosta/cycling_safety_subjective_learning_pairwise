import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from timeit import default_timer as timer


class ComparisonsDataset(Dataset):
    """Cycling Safety Perception dataset."""

    def __init__(self, dataframe, root_dir, transform=None, logger=None,):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with comparisons images and scores.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            logger (logging, optional): Logger object
        """
        self.comparisons_frame = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.logger = logger

    def __len__(self):
        return len(self.comparisons_frame)

    def __getitem__(self, idx):
        start = timer()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get left and right image paths
        img_l_name = os.path.join(self.root_dir,
                                  self.comparisons_frame.iloc[idx]['image_l'])
        img_r_name = os.path.join(self.root_dir,
                                  self.comparisons_frame.iloc[idx]['image_r'])

        # Open left and right images
        image_l = Image.open(img_l_name)
        image_r = Image.open(img_r_name)

        # Get scores, aka label for ranking
        score = self.comparisons_frame.iloc[idx]['score']
        score = score.astype('int')

        # Get scores, aka label, for classification
        score_classification = self.comparisons_frame.iloc[idx]['score_classification']
        score_classification = score_classification.astype('int')
        
        sample = {'image_l': image_l, 
                  'image_r': image_r, 
                  'score_r': score,
                  'score_c': score_classification,
                  'image_l_name': img_l_name,
                  'image_r_name': img_r_name,
                  }
        
        if self.transform:
            sample = self.transform(sample)

        end = timer()
        if self.logger:
            self.logger.info(f'DATALOADER, {end-start:.4f}')

        return sample


class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image_l, image_r = sample['image_l'], sample['image_r']

        return {'image_l': self.transform(image_l),
                'image_r': self.transform(image_r),
                'score_r': sample['score_r'],
                'score_c': sample['score_c'],
                'image_l_name': sample['image_l_name'],
                'image_r_name': sample['image_r_name'],
                }
