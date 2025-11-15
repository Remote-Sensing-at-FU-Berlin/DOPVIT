import os
import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import torch.nn.functional as F
from .transforms import get_resize, to_tensor

class TIFDataset(Dataset):
    """Dataset that reads labelled TIFFs.
    Expects a label file where each line contains: <filename> <label>
    (a header line is allowed and skipped).
    """

    def __init__(self, image_dir, label_file, transform=None, return_extra=False, expected_channels=4, crop_size=(50,50)):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.return_extra = return_extra
        self.expected_channels = expected_channels
        self.crop_size = crop_size

        # load labels
        self.labels = {}
        with open(label_file, 'r') as f:
            first = f.readline()
            # naive header detection: if first token is not a path, keep it but also treat as header
            if len(first.strip().split()) >= 2 and any(c.isalpha() for c in first):
                # assume header; continue reading
                pass
            else:
                # If first looks like data, parse it
                f.seek(0)
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = os.path.basename(parts[0]).strip('"')
                    try:
                        self.labels[filename] = int(parts[1])
                    except ValueError:
                    # skip malformed label
                    continue
    
        all_images = sorted(glob.glob(str(self.image_dir / '*.tif')))
        self.image_files = [p for p in all_images if os.path.basename(p) in self.labels]
    
        if len(self.image_files) == 0:
            print("Warning: No images matched labels. Example labels:", list(self.labels.keys())[:5])
    
    def __len__(self):
            return len(self.image_files)


    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = tifffile.imread(img_path).astype(np.float32)
        img = np.nan_to_num(img, nan=0.0)
        # Ensure channel-last image
        if img.ndim == 2:
            img = np.stack([img]*self.expected_channels, axis=-1)
        elif img.shape[-1] == 3 and self.expected_channels == 4:
            h, w = img.shape[:2]
            img = np.dstack([img, np.zeros((h, w), dtype=np.float32)])
            
        # to tensor CHW
        img_t = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        # center crop if needed
        c_h, c_w = self.crop_size
        _, h, w = img_t.shape
        if h < c_h or w < c_w:
            # pad symmetrically
            pad_h = max(0, c_h - h)
            pad_w = max(0, c_w - w)
            padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            img_t = F.pad(img_t, padding, value=0)
            _, h, w = img_t.shape
        top = (h - c_h) // 2
        left = (w - c_w) // 2
        img_t = img_t[:, top:top + c_h, left:left + c_w]
        
        if self.transform is not None:
            img_t = self.transform(img_t)
                
        base = os.path.basename(img_path)
        label = self.labels[base]
        
        if self.return_extra:
            return img_t, torch.tensor(label), img_path, label
        return img_t, torch.tensor(label)

class TransformingSubset(torch.utils.data.Dataset):
    """Wrap a Subset to apply transforms lazily and optionally return extra metadata."""
    
    def __init__(self, subset, transform=None, return_extra=False, resize=(224,224)):
        self.subset = subset
        self.transform = transform
        self.return_extra = return_extra
        self.resize = get_resize(resize)

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        sample = self.subset[idx]
        if self.return_extra and len(sample) >= 4:
            img, label, path, orig = sample
        else:
            img, label = sample
            path, orig = None, None
    
    
        img = self.resize(img)
        if self.transform:
            img = self.transform(img)
        if self.return_extra:
            return img, label, path, orig
        return img, label
