
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_download import BlockDataset

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    transform: transforms.Compose, 
    batch_size: int, 
    block_size: int,
    target:int,
    num_workers: int=0
):

  #GAUSIAN NOISE AUGUMENT -> TRUE else AUGUMENT -> FALSE
  train_data = BlockDataset("../../data/pretrained/tens_sequence/tens_concatenated.txt",block_size, augment=False)
  test_data = BlockDataset("../../data/pretrained/tens_sequence/tens_test.txt",block_size, augment=False)
  
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, 
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader