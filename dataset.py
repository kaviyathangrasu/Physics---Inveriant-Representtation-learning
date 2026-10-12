import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class S2LookingDataset(Dataset):

    def __init__(
        self,
        base_path,
        split="train",
        image_size=256
    ):

        self.base_path = base_path
        self.split = split

        # -----------------------------------------
        # Dataset directories
        # -----------------------------------------
        self.image1_dir = os.path.join(
            base_path, split, "Image1"
        )

        self.image2_dir = os.path.join(
            base_path, split, "Image2"
        )

        self.label_dir = os.path.join(
            base_path, split, "label"
        )

        self.label1_dir = os.path.join(
            base_path, split, "label1"
        )

        self.label2_dir = os.path.join(
            base_path, split, "label2"
        )

        # -----------------------------------------
        # Image preprocessing
        # -----------------------------------------
        self.image_transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # -----------------------------------------
        # Label preprocessing
        # -----------------------------------------
        self.label_transform = transforms.Compose([

            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST
            ),

            transforms.ToTensor()
        ])

        # -----------------------------------------
        # Get common filenames
        # -----------------------------------------
        image1_files = {
            f for f in os.listdir(self.image1_dir)
            if f.endswith(".png")
        }

        image2_files = {
            f for f in os.listdir(self.image2_dir)
            if f.endswith(".png")
        }

        label_files = {
            f for f in os.listdir(self.label_dir)
            if f.endswith(".png")
        }

        self.files = sorted(
            image1_files
            & image2_files
            & label_files
        )

        print()
        print(f"{split.upper()} DATASET")
        print("--------------------------------------")
        print("Image1 files :", len(image1_files))
        print("Image2 files :", len(image2_files))
        print("Label files  :", len(label_files))
        print("Common files :", len(self.files))

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        # -----------------------------------------
        # Paths
        # -----------------------------------------
        image1_path = os.path.join(
            self.image1_dir,
            filename
        )

        image2_path = os.path.join(
            self.image2_dir,
            filename
        )

        label_path = os.path.join(
            self.label_dir,
            filename
        )

        # -----------------------------------------
        # Load images
        # -----------------------------------------
        image1 = Image.open(
            image1_path
        ).convert("RGB")

        image2 = Image.open(
            image2_path
        ).convert("RGB")

        label = Image.open(
            label_path
        ).convert("L")

        # -----------------------------------------
        # Transform images
        # -----------------------------------------
        image1 = self.image_transform(image1)
        image2 = self.image_transform(image2)

        # -----------------------------------------
        # Transform label
        # -----------------------------------------
        label = self.label_transform(label)

        # Binary mask
        label = (label > 0.5).float()

        return image1, image2, label
