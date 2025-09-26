import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import collections
import os 
from utils.concept1_utils.utils import clip_image, denormalize_image, BNFeatureHook, lr_cosine_policy, save_images

def infer(model_lists, ipc_id, num_class, dataset, iteration, lr, batch_size, init_path, ipc_init, store_best_images):
    print("get_images call")
    save_every = 100
    batch_size = batch_size
    best_cost = 1e4

    loss_packed_features = [
        [BNFeatureHook(module) for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        for model in model_lists
    ]

    if len(loss_packed_features) > 2 and len(loss_packed_features[2]) > 1:
        loss_packed_features[2].pop(1)  

    targets_all = torch.LongTensor(np.arange(num_class))


    for kk in range(0, num_class, batch_size):
        targets = targets_all[kk : min(kk + batch_size, num_class)].to("cuda")

        model_index = ipc_id // ipc_init - 1
        model_teacher = model_lists[model_index]
        loss_r_feature_layers = loss_packed_features[model_index]

        # initialization
        init_file = f"{init_path}/tensor_{ipc_id % ipc_init}.pt"
        if os.path.exists(init_file):
            loaded_tensor = torch.load(init_file).clone()
            input_original = loaded_tensor.to("cuda").detach()
        else:
            rand_idx = random.randint(0, len(dataset) - 1)
            img, _ = dataset[rand_idx]
            input_original = img.unsqueeze(0).to("cuda").detach()
            
        uni_perb = torch.zeros_like(input_original, requires_grad=True, device="cuda")

        iterations_per_layer = iteration if ipc_id >= ipc_init else 0
        inputs = input_original if iterations_per_layer == 0 else input_original + uni_perb

        optimizer = optim.Adam([uni_perb], lr=lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(lr, 0, iterations_per_layer)
        criterion = nn.CrossEntropyLoss().cuda()          

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)
            inputs = input_original + uni_perb

            off1, off2 = random.randint(0, jitter), random.randint(0, jitter)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)

            # R_cross classification loss
            loss_ce = criterion(outputs, targets)

            # R_feature loss
            rescale = [first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers) - 1)]

            loss_r_bn_feature = [
                mod.r_feature.to(loss_ce.device) * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)
            ]
            loss_r_bn_feature = torch.stack(loss_r_bn_feature).sum()

            loss_aux = r_bn * loss_r_bn_feature

            loss = loss_ce + loss_aux

            if iteration % save_every == 0:
                print("------------iteration {}----------".format(iteration))
                print("loss_ce", loss_ce.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("loss_total", loss.item())
                # comment below line can speed up the training (no validation process)
                # if hook_for_display is not None:
                #     acc_jit, _ = hook_for_display(inputs_jit, targets)
                #     acc_image, loss_image = hook_for_display(inputs, targets)

                #     metrics = {
                #         'crop/acc_crop': acc_jit,
                #         'image/acc_image': acc_image,
                #         'image/loss_image': loss_image,
                #     }
                #     wandb_metrics.update(metrics)

                # metrics = {
                #     'crop/loss_ce': loss_ce.item(),
                #     'crop/loss_r_bn_feature': loss_r_bn_feature.item(),
                #     'crop/loss_total': loss.item(),
                # }
                # wandb_metrics.update(metrics)
                # wandb.log(wandb_metrics)

            # do image update
            loss.backward()

            optimizer.step()
            # clip color outlayers
            inputs.data = clip_image(inputs.data, dataset)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = denormalize_image(best_inputs, dataset)
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

    torch.cuda.empty_cache()