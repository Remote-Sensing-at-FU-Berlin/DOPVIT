import torchvision.transforms as T
import torch
import torch.nn.functional as F

def get_augmentations(cfg):
  aug_list = []
  if cfg.get('horizontal_flip', False):
    aug_list.append(T.RandomHorizontalFlip())
  if cfg.get('vertical_flip', False):
    aug_list.append(T.RandomVerticalFlip())
  if cfg.get('rotation', 0):
    aug_list.append(T.RandomRotation(cfg['rotation']))
  if cfg.get('affine'):
    a = cfg['affine']
    aug_list.append(T.RandomAffine(degrees=0, translate=tuple(a.get('translate', (0,0))), scale=tuple(a.get('scale', (1,1)))))
  
  # Note: torchvision transforms expect PIL images or tensors in [0,1]
  return T.Compose(aug_list) if aug_list else None
  
def get_resize(size=(224,224)):
  def _resize(tensor):
    # expects CHW tensor
    if not isinstance(tensor, torch.Tensor):
      tensor = torch.tensor(tensor)
    tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)
    return tensor
  return _resize

def to_tensor(x):
  # identity for already-tensor inputs
  if isinstance(x, torch.Tensor):
    return x
  return torch.tensor(x)
