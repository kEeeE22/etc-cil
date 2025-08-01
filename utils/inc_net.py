import copy
import logging
import torch
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear, TagFex_SimpleLinear
from convs.modified_represnet import resnet18_rep,resnet34_rep
from convs.resnet_cbam import resnet18_cbam,resnet34_cbam,resnet50_cbam
from convs.memo_resnet import  get_resnet18_imagenet as get_memo_resnet18 #for MEMO imagenet
from convs.memo_cifar_resnet import get_resnet32_a2fc as get_memo_resnet32 #for MEMO cifar
from convs.ACL_buffer import RandomBuffer, activation_t
from convs.linears import RecursiveLinear
from convs.etc_cnn import ETC_CNN
from typing import Dict, Any


def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained,args=args)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained,args=args)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained,args=args)
    elif name == "cosine_resnet18":
        return cosine_resnet18(pretrained=pretrained,args=args)
    elif name == "cosine_resnet32":
        return cosine_resnet32()
    elif name == "cosine_resnet34":
        return cosine_resnet34(pretrained=pretrained,args=args)
    elif name == "cosine_resnet50":
        return cosine_resnet50(pretrained=pretrained,args=args)
    elif name == "resnet18_rep":
        return resnet18_rep(pretrained=pretrained,args=args)
    elif name == "resnet18_cbam":
        return resnet18_cbam(pretrained=pretrained,args=args)
    elif name == "resnet34_cbam":
        return resnet34_cbam(pretrained=pretrained,args=args)
    elif name == "resnet50_cbam":
        return resnet50_cbam(pretrained=pretrained,args=args)
    
    # MEMO benchmark backbone
    elif name == 'memo_resnet18':
        _basenet, _adaptive_net = get_memo_resnet18()
        return _basenet, _adaptive_net
    elif name == 'memo_resnet32':
        _basenet, _adaptive_net = get_memo_resnet32()
        return _basenet, _adaptive_net
    elif name == 'etc_cnn':
        return ETC_CNN()
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        self.convnet.load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)  # phải trả về dict với key "features"
        logits = self.fc(x["features"])  # SimpleLinear
        out = {
            "logits": logits,
            "features": x["features"]
        }
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )

class IL2ANet(IncrementalNet):

    def update_fc(self, num_old, num_total, num_aux):
        fc = self.generate_fc(self.feature_dim, num_total+num_aux)
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_old] = weight[:num_old]
            fc.bias.data[:num_old] = bias[:num_old]
        del self.fc
        self.fc = fc

class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer_BIC(nn.Module):
    def __init__(self):
        super(BiasLayer_BIC, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer_BIC())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:

                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def regenerate_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        del self.fc
        self.fc = fc
        return fc

class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        # no need the convnet.

        print(
            'Clear the convnet in MultiBranchCosineIncrementalNet, since we are using self.convnets with dual branches')
        self.convnet = torch.nn.Identity()
        for param in self.convnet.parameters():
            param.requires_grad = False

        self.convnets = nn.ModuleList()
        self.args = args

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        # import pdb; pdb.set_trace()
        out = self.fc(features)
        out.update({"features": features})
        return out

    def construct_dual_branch_network(self, trained_model, tuned_model, cls_num):
        self.convnets.append(trained_model.convnet)
        self.convnets.append(tuned_model.convnet)

        self._feature_dim = self.convnets[0].out_dim * len(self.convnets)
        self.fc = self.generate_fc(self._feature_dim, cls_num)


class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc
    

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x , bias=True):
        ret_x = x.clone()
        ret_x = (self.alpha+1) * x # + self.beta
        if bias:
            ret_x = ret_x + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class BEEFISONet(nn.Module):
    def __init__(self, args, pretrained):
        super(BEEFISONet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.old_fc = None
        self.new_fc = None
        self.task_sizes = []
        self.forward_prototypes = None
        self.backward_prototypes = None
        self.args = args
        self.biases = nn.ModuleList()

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        
        if self.old_fc is None:
            fc = self.new_fc
            out = fc(features)
        else:
            '''
            merge the weights
            '''
            new_task_size = self.task_sizes[-1]
            fc_weight = torch.cat([self.old_fc.weight,torch.zeros((new_task_size,self.feature_dim-self.out_dim)).cuda()],dim=0)             
            new_fc_weight = self.new_fc.weight
            new_fc_bias = self.new_fc.bias
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([*[self.biases[i](self.backward_prototypes.weight[i].unsqueeze(0),bias=False) for _ in range(self.task_sizes[i])],new_fc_weight],dim=0)
                new_fc_bias = torch.cat([*[self.biases[i](self.backward_prototypes.bias[i].unsqueeze(0),bias=True) for _ in range(self.task_sizes[i])], new_fc_bias])
            fc_weight = torch.cat([fc_weight,new_fc_weight],dim=1)
            fc_bias = torch.cat([self.old_fc.bias,torch.zeros(new_task_size).cuda()])
            fc_bias=+new_fc_bias
            logits = features@fc_weight.permute(1,0)+fc_bias
            out = {"logits":logits}        

            new_fc_weight = self.new_fc.weight
            new_fc_bias = self.new_fc.bias
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([self.backward_prototypes.weight[i].unsqueeze(0),new_fc_weight],dim=0)
                new_fc_bias = torch.cat([self.backward_prototypes.bias[i].unsqueeze(0), new_fc_bias])
            out["train_logits"] = features[:,-self.out_dim:]@new_fc_weight.permute(1,0)+new_fc_bias 
        out.update({"eval_logits": out["logits"],"energy_logits":self.forward_prototypes(features[:,-self.out_dim:])["logits"]})
        return out

    def update_fc_before(self, nb_classes):
        new_task_size = nb_classes - sum(self.task_sizes)
        self.biases = nn.ModuleList([BiasLayer() for i in range(len(self.task_sizes))])
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        if self.new_fc is not None:
            self.fe_fc = self.generate_fc(self.out_dim, nb_classes)
            self.backward_prototypes = self.generate_fc(self.out_dim,len(self.task_sizes))
            self.convnets[-1].load_state_dict(self.convnets[0].state_dict())
        self.forward_prototypes = self.generate_fc(self.out_dim, nb_classes)
        self.new_fc = self.generate_fc(self.out_dim,new_task_size)
        self.task_sizes.append(new_task_size)
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc
    
    def update_fc_after(self):
        if self.old_fc is not None:
            old_fc = self.generate_fc(self.feature_dim, sum(self.task_sizes))
            new_task_size = self.task_sizes[-1]
            old_fc.weight.data = torch.cat([self.old_fc.weight.data,torch.zeros((new_task_size,self.feature_dim-self.out_dim)).cuda()],dim=0)             
            new_fc_weight = self.new_fc.weight.data
            new_fc_bias = self.new_fc.bias.data
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([*[self.biases[i](self.backward_prototypes.weight.data[i].unsqueeze(0),bias=False) for _ in range(self.task_sizes[i])], new_fc_weight],dim=0)
                new_fc_bias = torch.cat([*[self.biases[i](self.backward_prototypes.bias.data[i].unsqueeze(0),bias=True) for _ in range(self.task_sizes[i])], new_fc_bias])
            old_fc.weight.data = torch.cat([old_fc.weight.data,new_fc_weight],dim=1)
            old_fc.bias.data = torch.cat([self.old_fc.bias.data,torch.zeros(new_task_size).cuda()])
            old_fc.bias.data+=new_fc_bias
            self.old_fc = old_fc
        else:
            self.old_fc  = self.new_fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma


class AdaptiveNet(nn.Module):
    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.TaskAgnosticExtractor , _ = get_convnet(args, pretrained) #Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList() #Specialized Blocks
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []
        self.args=args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.AdaptiveExtractors)
    
    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out=self.fc(features) #{logits: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"] 

        out.update({"aux_logits":aux_logits,"features":features})
        out.update({"base_features":base_feature_map})
        return out
                
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''
        
    def update_fc(self,nb_classes):
        _ , _new_extractor = get_convnet(self.args)
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            logging.info(self.AdaptiveExtractors[-1])
            self.out_dim=self.AdaptiveExtractors[-1].feature_dim        
        fc = self.generate_fc(self.feature_dim, nb_classes)             
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)
 
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['convnet']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k:v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k:v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class ACILNet(BaseNet):
    """
    Network structure of the ACIL [1].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
        Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
    """
    def __init__(
        self,
        args: Dict[str, Any],
        buffer_size: int = 8192,
        gamma: float = 0.1,
        pretrained: bool = False,
        device=None,
        dtype=torch.double,
    ) -> None:
        super().__init__(args, pretrained)
        assert isinstance(
            self.convnet, torch.nn.Module
        ), "The backbone network `convnet` must be a `torch.nn.Module`."
        self.convnet: torch.nn.Module = self.convnet.to(device, non_blocking=True)

        self.args = args
        self.buffer_size: int = buffer_size
        self.gamma: float = gamma
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        X = self.convnet(X)["features"]
        X = self.buffer(X)
        X = self.fc(X)["logits"]
        return {"logits": X}

    def update_fc(self, nb_classes: int) -> None:
        self.fc.update_fc(nb_classes)

    def generate_fc(self, *_) -> None:
        self.fc = RecursiveLinear(
            self.buffer_size,
            self.gamma,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

    def generate_buffer(self) -> None:
        self.buffer = RandomBuffer(
            self.feature_dim, self.buffer_size, device=self.device, dtype=self.dtype
        )

    def after_task(self) -> None:
        self.fc.after_task()

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = self.convnet(X)["features"]
        X = self.buffer(X)
        Y: torch.Tensor = torch.nn.functional.one_hot(y, self.fc.out_features)
        self.fc.fit(X, Y)


class DSALNet(ACILNet):
    """
    Network structure of the DS-AL [1].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
    """
    def __init__(
        self,
        args: Dict[str, Any],
        buffer_size: int = 8192,
        gamma_main: float = 1e-3,
        gamma_comp: float = 1e-3,
        C: float = 1,
        activation_main: activation_t = torch.relu,
        activation_comp: activation_t = torch.tanh,
        pretrained: bool = False,
        device=None,
        dtype=torch.double,
    ) -> None:
        self.C = C
        self.gamma_comp = gamma_comp
        self.activation_main = activation_main
        self.activation_comp = activation_comp
        super().__init__(args, buffer_size, gamma_main, pretrained, device, dtype)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        X = self.buffer(self.convnet(X)["features"])
        X_main = self.fc(self.activation_main(X))["logits"]
        X_comp = self.fc_comp(self.activation_comp(X))["logits"]
        return {"logits": X_main + self.C * X_comp}

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        num_classes = max(self.fc.out_features, int(y.max().item()) + 1)
        Y_main = torch.nn.functional.one_hot(y, num_classes=num_classes)
        X = self.buffer(self.convnet(X)["features"])

        # Train the main stream
        X_main = self.activation_main(X)
        self.fc.fit(X_main, Y_main)
        self.fc.after_task()

        # Previous label cleansing (PLC)
        Y_comp = Y_main - self.fc(X_main)["logits"]
        Y_comp[:, : -self.increment_size] = 0

        # Train the compensation stream
        X_comp = self.activation_comp(X)
        self.fc_comp.fit(X_comp, Y_comp)

    @torch.no_grad()
    def after_task(self) -> None:
        self.fc.after_task()
        self.fc_comp.after_task()

    def generate_buffer(self) -> None:
        self.buffer = RandomBuffer(
            self.feature_dim,
            self.buffer_size,
            activation=None,
            device=self.device,
            dtype=self.dtype,
        )

    def generate_fc(self, *_) -> None:
        # Main stream
        self.fc = RecursiveLinear(
            self.buffer_size,
            self.gamma,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

        # Compensation stream
        self.fc_comp = RecursiveLinear(
            self.buffer_size,
            self.gamma_comp,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

    def update_fc(self, nb_classes) -> None:
        self.increment_size = nb_classes - self.fc.out_features
        self.fc.update_fc(nb_classes)
        self.fc_comp.update_fc(nb_classes)

class TagFexNet(nn.Module):
    def __init__(self, args, pretrained):
        super(TagFexNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

        self._device = args["device"][0]
        self.ta_net = get_convnet(args).to(self._device)

        if hasattr(self.ta_net, 'fc'):
            self.ta_net.fc = None
        self.ts_attn = None
        self.trans_classifier = None

        self.projector = nn.Sequential(
            TagFex_SimpleLinear(self.ta_feature_dim, self.args["proj_hidden_dim"]),
            nn.ReLU(True),
            TagFex_SimpleLinear(self.args["proj_hidden_dim"], self.args["proj_output_dim"]),
        ).to(self._device)
        self.predictor = None

    @property
    def ta_feature_dim(self):
        return self.ta_net.out_dim
    
    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        x.to(self._device)
        ts_outs = [convnet(x) for convnet in self.convnets]
        features = [ts_out["features"] for ts_out in ts_outs]
        features = torch.cat(features, 1)

        #out = self.fc(features)  # {logics: self.fc(features)}
        out = {}
        out["logits"] = self.fc(features)
        aux_logits = self.aux_fc(features[:, -self.out_dim :])

        out.update({"aux_logits": aux_logits, "features": features})
        
        ta_fmap = self.ta_net(x)['fmaps'][-1] # (bs, C, H, W)
        ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1) # (bs, H*W, C) -mean-> (bs, C)
        #print(ta_feature)
        #assert 0
        embedding = self.projector(ta_feature)

        out.update({
                'ta_feature': ta_feature,
                'embedding': embedding,
        })

        if self.trans_classifier is not None:
            ts_feature = ts_outs[-1]["fmaps"][-1].flatten(2).permute(0, 2, 1)
            ta_features = ta_fmap.flatten(2).permute(0, 2, 1)# (bs, H*W, C) 
            merged_feature = self.ts_attn(ta_features.detach(), ts_feature).mean(1)
            trans_logits = self.trans_classifier(merged_feature)
            out.update(trans_logits=trans_logits)

        if self.predictor is not None:
            predicted_feature = self.predictor(ta_feature)
            out.update(predicted_feature=predicted_feature)            
        
        return out
        """
        {
            'ta_feature': ta_feature,
            'embedding': embedding,
            'trans_logits': trans_logits,
            'predicted_feature': predicted_feature,
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """
     
    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args).to(self._device))
        else:
            self.convnets.append(get_convnet(self.args).to(self._device))
            init_interpolation_factor = self.args["init_interpolation_factor"]
            for ts_old_parameter, ts_new_parameter, ta_prarameter in zip(self.convnets[-2].parameters(), self.convnets[-1].parameters(), self.ta_net.parameters()):
                ts_new_parameter.data = init_interpolation_factor * ts_old_parameter.data + (1 - init_interpolation_factor) * ta_prarameter.data

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1).to(self._device)

        """new added"""
        if self.predictor is None:
            self.predictor = self.generate_fc(self.ta_feature_dim, self.ta_feature_dim).to(self._device)
        if self.ts_attn is None:
            self.ts_attn = TSAttention(self.out_dim, self.args['attn_num_heads'], device=self._device)##
        else:
            self.ts_attn._reset_parameters()
            # trans_classifier is the merged classifier
        self.trans_classifier = self.generate_fc(self.ta_net.out_dim, new_task_size).to(self._device)

    def generate_fc(self, in_dim, out_dim):
        fc = TagFex_SimpleLinear(in_dim, out_dim).to(self._device)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    def get_freezed_copy_ta(self):
        from copy import deepcopy
        ta_net_copy = deepcopy(self.ta_net)
        for p in ta_net_copy.parameters():
            p.requires_grad_(False)
        return ta_net_copy.eval()

    def get_freezed_copy_projector(self):
        from copy import deepcopy
        projector_copy = deepcopy(self.projector)
        for p in projector_copy.parameters():
            p.requires_grad_(False)
        return projector_copy.eval()
    
    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class TSAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm_ts = nn.LayerNorm(embed_dim, device=device)
        self.norm_ta = nn.LayerNorm(embed_dim, device=device)

        self.weight_q = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_k_ts = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_k_ta = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_v_ts = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
        self.weight_v_ta = nn.Parameter(torch.empty((embed_dim, embed_dim), device=device))
    
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.weight_q)
        nn.init.xavier_normal_(self.weight_k_ts)
        nn.init.xavier_normal_(self.weight_k_ta)
        nn.init.xavier_normal_(self.weight_v_ts)
        nn.init.xavier_normal_(self.weight_v_ta)

        self.norm_ta.reset_parameters()
        self.norm_ts.reset_parameters()
    
    def forward(self, ta_feats, ts_feats):
        bs, N, C = ta_feats.shape
        # feats: (bs, N, C)
        ta_feats = self.norm_ta(ta_feats)
        ts_feats = self.norm_ts(ts_feats)

        q = (ts_feats @ self.weight_q).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2) # (bs, H, N, Ch)
        k_ts = (ts_feats @ self.weight_k_ts).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k_ta = (ta_feats @ self.weight_k_ta).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_ts = (ts_feats @ self.weight_v_ts).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_ta = (ta_feats @ self.weight_v_ta).reshape(bs, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        feat = nn.functional.scaled_dot_product_attention(q, torch.cat((k_ta, k_ts), dim=2), torch.cat((v_ta, v_ts), dim=2)) # (bs, H, N, Ch) # use default scale

        feat = feat.transpose(1, 2).flatten(2)

        return feat