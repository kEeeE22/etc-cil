import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn

def clip_image(image_tensor, dataset: str) -> torch.Tensor:
    """
    adjust the input w.r.t mean and variance
    """

    if dataset == 'cifar100':
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
    elif dataset == 'cifar10':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
    elif dataset == 'tinyIN':
        mean = np.array([0.4802, 0.4481, 0.3975])
        std = np.array([0.2302, 0.2265, 0.2262])
    elif dataset == 'etc_256':
        mean = np.array([0.5])
        std = np.array([0.5])
    n_channels = image_tensor.shape[1]
    mean = torch.tensor(mean, device=image_tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=image_tensor.device).view(1, -1, 1, 1)

    min_val = (-mean / std).expand_as(image_tensor)
    max_val = ((1 - mean) / std).expand_as(image_tensor)

    image_tensor = torch.max(torch.min(image_tensor, max_val), min_val)
    return image_tensor


def denormalize_image(image_tensor, dataset: str) -> torch.Tensor:
    """
    reconvert floats back to input range
    """
    if dataset == 'cifar100':
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
    elif dataset == 'cifar10':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
    elif dataset == 'tinyIN':
        mean = np.array([0.4802, 0.4481, 0.3975])
        std = np.array([0.2302, 0.2265, 0.2262])
    elif dataset == 'etc_256':
        mean = np.array([0.5])
        std = np.array([0.5])
    n_channels = image_tensor.shape[1]
    mean = torch.tensor(mean, device=image_tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=image_tensor.device).view(1, -1, 1, 1)

    min_val = (-mean / std).expand_as(image_tensor)
    max_val = ((1 - mean) / std).expand_as(image_tensor)

    image_tensor = torch.max(torch.min(image_tensor, max_val), min_val)
    return image_tensor


class BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def save_images(syn_data_path, images, targets, ipc_id):
    for id in range(images.shape[0]):
        # Xác định class ID
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        # Tạo thư mục lưu
        dir_path = f'{syn_data_path}/new{class_id:03d}'
        os.makedirs(dir_path, exist_ok=True)
        place_to_store = f'{dir_path}/class{class_id:03d}_id{ipc_id:03d}.jpg'

        # Lấy ảnh
        image_np = images[id].data.cpu().numpy()

        # Nếu có 3 chiều (C, H, W)
        if image_np.ndim == 3:
            image_np = np.transpose(image_np, (1, 2, 0))

            # Nếu chỉ 1 kênh, chuyển về 2D (H, W)
            if image_np.shape[2] == 1:
                image_np = image_np.squeeze(2)
                mode = "L"  # grayscale
            elif image_np.shape[2] == 3:
                mode = "RGB"
            else:
                raise ValueError(f"Unsupported channel count: {image_np.shape[2]}")
        elif image_np.ndim == 2:
            mode = "L"
        else:
            raise ValueError(f"Unexpected image shape: {image_np.shape}")

        # Chuẩn hóa về [0,255]
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

        # Tạo ảnh PIL đúng mode
        pil_image = Image.fromarray(image_np, mode=mode)
        pil_image.save(place_to_store)

def save_images_ufc(syn_data_path, images, targets, ipc_id, model_index):
    for id in range(90):#(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(syn_data_path):
            os.mkdir(syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(syn_data_path, class_id)
        place_to_store = dir_path + '/class{:03d}_model{:03d}_id{:03d}.jpg'.format(class_id,model_index, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)

from PIL import Image
from torchvision import transforms
import numpy as np
import os

def load_synthetic_dataset_as_numpy(syn_root="./syn", dataset_name="etc_256"):
    """
    Load all synthetic JPG images (distilled samples) from disk as numpy arrays.
    Returns (images_np, targets_np)
    """
    if not os.path.exists(syn_root):
        print(f"[WARN] Synthetic folder '{syn_root}' not found.")
        return None

    images, targets = [], []
    transform = transforms.ToTensor()
    print(f"📦 Loading synthetic data from {syn_root} ...")

    for class_dir in sorted(os.listdir(syn_root)):
        class_path = os.path.join(syn_root, class_dir)
        if not os.path.isdir(class_path):
            continue
        try:
            class_id = int(class_dir.replace("new", ""))
        except:
            continue

        for fname in os.listdir(class_path):
            if not fname.endswith(".jpg"):
                continue
            img_path = os.path.join(class_path, fname)
            img = Image.open(img_path).convert("L" if dataset_name == "etc_256" else "RGB")
            img_tensor = transform(img)
            images.append(img_tensor.numpy())
            targets.append(class_id)

    if len(images) == 0:
        print("[INFO] No synthetic samples found.")
        return None

    images_np = np.stack(images)
    targets_np = np.array(targets)
    print(f"✅ Loaded {len(images_np)} synthetic images across {len(np.unique(targets_np))} classes.")
    return images_np, targets_np
