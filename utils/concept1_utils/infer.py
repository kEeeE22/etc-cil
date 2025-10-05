import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import collections
import os 
from utils.concept1_utils.utils import clip_image, denormalize_image, BNFeatureHook, lr_cosine_policy, save_images


jitter = 4
first_bn_multiplier = 10.0
r_bn = 1

def infer_gen(model_lists, ipc_id, num_class, dataset, iteration, lr, batch_size, init_path, ipc_init, known_classes, store_best_images):
    print("get_images call")
    save_every = 100
    best_cost = 1e4
    syn = []   
    ufc = []
    loss_packed_features = [
        [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        for model in model_lists
    ]

    if len(loss_packed_features) > 2 and len(loss_packed_features[2]) > 1:
        loss_packed_features[2].pop(1)  

    targets_all = torch.LongTensor(np.arange(num_class))

    for class_id in range(num_class):
        targets = torch.LongTensor([class_id]).to("cuda")
        if len(model_lists) == 1:
            model_index = 0
        elif len(model_lists) == 2:
            half = ipc_init
            model_index = 0 if ipc_id < half else 1
        else:
            # fallback an toàn
            model_index = min(ipc_id // ipc_init, len(model_lists) - 1)

        model_teacher = model_lists[model_index]
        loss_r_feature_layers = loss_packed_features[model_index]

        # initialization
        is_old_class = class_id < known_classes
        if is_old_class:
            init_file = f"{init_path}/tensor_class{class_id:03d}.pt"
            if os.path.exists(init_file):
                loaded_tensor = torch.load(init_file).clone()
                input_original = loaded_tensor.to("cuda").detach()
                print(f"[OLD] Loaded init tensor for class {class_id} from {init_file}")
            else:
                print(f"[WARN] Missing tensor for class {class_id}, fallback random init")
                rand_idx = random.randint(0, len(dataset) - 1)
                _, img, _ = dataset[rand_idx]
                input_original = img.unsqueeze(0).to("cuda").detach()
        else:
            rand_idx = random.randint(0, len(dataset) - 1)
            _, img, _ = dataset[rand_idx]
            input_original = img.unsqueeze(0).to("cuda").detach()
            print(f"[NEW] Random init for class {class_id}")
            
        uni_perb = torch.zeros_like(input_original, requires_grad=True, device="cuda")

        iterations_per_layer = iteration if ipc_id >= ipc_init else 0
        inputs = input_original if iterations_per_layer == 0 else input_original + uni_perb

        optimizer = optim.Adam([uni_perb], lr=lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(lr, 0, iterations_per_layer)
        criterion = nn.CrossEntropyLoss().cuda()          

        best_inputs = None

        for it in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, it, it)
            inputs = input_original + uni_perb

            off1, off2 = random.randint(0, jitter), random.randint(0, jitter)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)["logits"]

            # classification loss
            loss_ce = criterion(outputs, targets)

            # BN feature loss
            rescale = [first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers) - 1)]
            loss_r_bn_feature = [
                mod.r_feature.to(loss_ce.device) * rescale[idx]
                for (idx, mod) in enumerate(loss_r_feature_layers)
            ]
            loss_r_bn_feature = torch.stack(loss_r_bn_feature).sum()

            loss_aux = r_bn * loss_r_bn_feature
            loss = loss_ce + loss_aux

            if it % save_every == 0:
                print(f"---- iteration {it} ----")
                print("loss_ce", loss_ce.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("loss_total", loss.item())

            # update
            loss.backward()
            optimizer.step()
            print(inputs.data.shape)
            inputs.data = clip_image(inputs.data, 'etc_256')

            if best_cost > loss.item() or it == 1:
                best_cost = loss.item()
                best_inputs = inputs.data.clone()

        if best_inputs is not None:
            best_inputs = denormalize_image(best_inputs, 'etc_256')
            syn.append((best_inputs.cpu(), targets.cpu()))

            if store_best_images:
                save_images(init_path, best_inputs, targets, ipc_id)

        optimizer.state = collections.defaultdict(dict)

    torch.cuda.empty_cache()
    return syn, ufc 
