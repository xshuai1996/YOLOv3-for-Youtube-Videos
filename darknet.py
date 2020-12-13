import torch.nn as nn
import torch
from utils import parse_config, download_pretrained_weights
import numpy as np


class RouteLayer(nn.Module):
    """
    Route layer concatenates output tensors of two specified layers, i.e. concatenate
    a*b*c and a*b*d to a*b*(c+d). If only one layer is given, then it outputs the output
    of this layer directly.
    """
    def __init__(self):
        super(RouteLayer, self).__init__()
        pass

    def forward(self, x):
        layer1_output, layer2_output = x
        if layer2_output is None:
            x = layer1_output
        else:
            x = torch.cat((layer1_output, layer2_output), dim=1)  # dim=1: channel dim
        return x


class ShortcutLayer(nn.Module):
    """
    Shortcut layer element-wise adds output tensors of two layers with same shape
    """
    def __init__(self):
        super(ShortcutLayer, self).__init__()
        pass

    def forward(self, x):
        layer1_output, layer2_output = x
        x = layer1_output + layer2_output
        return x


class DetectionLayer(nn.Module):
    def __init__(self, anchors, img_size, num_classes):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.img_size = img_size
        self.bbox_attrs = num_classes + 5  # YOLOv3 have 5 basic attributes (i.e. x, y, w, h, confidence) plus # classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_anchors = len(self.anchors)
        self.init_stride_grid = False   # only need to specify stride and grid size once
        self.stride = None
        self.grid_size = None
        self.x_y_offset = None


    def forward(self, x):
        # specify the grid size and stride
        if not self.init_stride_grid:
            self.stride = self.img_size // x.size(2)
            self.grid_size = self.img_size // self.stride

            # resize the anchor boxes and repeat for each cell
            self.anchors = torch.FloatTensor([(a[0] / self.stride, a[1] / self.stride) for a in self.anchors]).to(self.device)
            # unsqueeze returns a new tensor with a dimension of size one inserted at the specified position. The
            # returned tensor shares the same underlying data with this tensor.
            self.anchors = self.anchors.repeat(self.grid_size * self.grid_size, 1).unsqueeze(0)

            # Add the offsets (i.e. the top-left co-ordinates of the grid) for (x, y)
            grid = np.arange(self.grid_size)
            a, b = np.meshgrid(grid, grid)
            x_offset = torch.FloatTensor(a).view(-1, 1).to(self.device)
            y_offset = torch.FloatTensor(b).view(-1, 1).to(self.device)
            self.x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, self.num_anchors).view(-1, 2).unsqueeze(0)

            self.init_stride_grid = True

        # reshape the tensor from (batch_size, channels, height, width) to (batch_size, height x width x num_anchor_box,
        # num_attribute_of_each_box)
        x = x.view(x.shape[0], self.bbox_attrs * self.num_anchors, self.grid_size * self.grid_size)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], self.grid_size * self.grid_size * self.num_anchors, self.bbox_attrs)

        # Sigmoid the x, y, confidence to make sure that (x, y) will not go out of the cell (bounded by sigmoid now),
        # and confidence score will be in range (0, 1)
        x[:, :, 0] = torch.sigmoid(x[:, :, 0])
        x[:, :, 1] = torch.sigmoid(x[:, :, 1])
        x[:, :, 4] = torch.sigmoid(x[:, :, 4])

        # change (x, y) to real location, which means adding the top-left co-ordinates of the grid
        x[:, :, :2] += self.x_y_offset
        # use exponential func for (w, h) to ensure they > 0. It's also ad-hoc, and experiments show that linear form
        # decreases the stability of model (claimed in paper)
        x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * self.anchors
        # basically same as confidence score, use sigmoid for all P of classes
        x[:, :, 5:] = torch.sigmoid((x[:, :, 5:]))
        # rescale the box to actual size
        x[:, :, :4] *= self.stride
        return x


def create_module_list(config, device):
    """
    Build module list according to configuration.
    :param config: parsed config
    :return net_params: general parameters of the network
    :return module_list: nn.ModuleList that contains all modules
    """
    net_params = config[0]
    module_list = nn.ModuleList()
    prev_out_channel = 3        # later conv layer need to specify in_channels. 3 for initial RGB
    output_channels = []        # keep record of # output channels for route and shortcut layers
    for ind, block in enumerate(config[1:]):
        module = nn.Sequential()

        # for conv layers
        if block['type'] == "convolutional":
            activation = block["activation"]
            batch_normalize = int(block["batch_normalize"]) if "batch_normalize" in block else 0
            bias = False if "batch_normalize" in block else True
            output_channel = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            pad = (kernel_size - 1) // 2 if int(block["pad"]) else 0

            # add conv layer
            conv = nn.Conv2d(in_channels=prev_out_channel, out_channels=output_channel, kernel_size=kernel_size,
                             stride=stride, padding=pad, bias=bias)
            module.add_module("conv_{}".format(ind), conv)

            # add batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(output_channel)
                module.add_module("batch_norm_{}".format(ind), bn)

            # add activation layer
            if activation == "leaky":
                activ = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                module.add_module("leaky_{0}".format(ind), activ)
            elif activation == "linear":
                pass
            else:
                raise Exception("Activation function {} not implemented. Please add a few lines here.".format(activation))

        # for upsampling layer
        elif block['type'] == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
            module.add_module("upsample_{}".format(ind), upsample)

        # for route layer. Route layer concatenates output tensors of two specified layers,
        # i.e. concatenate a*b*c and a*b*d to a*b*(c+d). If only one layer is given, output
        # this layer directly
        elif block["type"] == "route":
            block["layers"] = block["layers"].split(',')
            block["layers"] = [int(layer) for layer in block["layers"]]

            # unify to absolute annotation (not relative location)
            if block["layers"][0] < 0:
                block["layers"][0] += ind
            if len(block["layers"]) > 1 and block["layers"][1] < 0:
                # if two layers are given
                block["layers"][1] += ind

            route = RouteLayer()
            module.add_module("route_{}".format(ind), route)

            if len(block["layers"]) == 1:
                output_channel = output_channels[block["layers"][0]]
            else:
                output_channel = output_channels[block["layers"][0]] + output_channels[block["layers"][1]]

        # for shortcut layer. Shortcut layer element-wise adds output tensors of two layers with same shape
        elif block["type"] == "shortcut":
            block["from"] = int(block["from"])
            activation = block["activation"]
            shortcut = ShortcutLayer()
            module.add_module("shortcut_{}".format(ind), shortcut)

            # unify to absolute annotation (not relative location)
            if block["from"] < 0:
                block["from"] += ind

            # add activation layer
            if activation == "leaky":
                activ = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                module.add_module("leaky_{0}".format(ind), activ)
            elif activation == "linear":
                pass
            else:
                raise Exception("Activation function {} not implemented. Please add a few lines here.".format(activation))

        # for YOLO layer
        elif block["type"] == "yolo":
            # there are 3 different scale for YOLOv3. each scale will use 3 anchor boxes, which are specified by mask
            mask = [int(x) for x in block["mask"].split(",")]
            block["anchors"] = [int(a) for a in block["anchors"].split(",")]
            block["anchors"] = [(block["anchors"][i], block["anchors"][i+1]) for i in range(0, len(block["anchors"]), 2)]
            # not clear
            anchors = [block["anchors"][i] for i in mask]
            detection = DetectionLayer(anchors, img_size=int(net_params['height']), num_classes=int(block["classes"]))
            module.add_module("Detection_{}".format(ind), detection)

        module_list.append(module)
        prev_out_channel = output_channel
        output_channels.append(output_channel)

    module_list.to(device)
    return net_params, module_list


class Darknet(nn.Module):
    """
    Define the network architecture of YOLOv3.
    """
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = parse_config(cfgfile)
        self.net_params, self.module_list = create_module_list(self.config, self.device)

    def forward(self, x):
        x = x.to(self.device)
        # store the output for route and shortcut layers, key, value are index and output tensor of layer individually
        outputs = {}
        pred_of_all_scale = []  # to store the prediction in different scale, and concatenate them later
        for ind, module in enumerate(self.config[1:]):
            if module['type'] == "convolutional":
                x = self.module_list[ind](x)

            elif module['type'] == "upsample":
                x = self.module_list[ind](x)

            elif module["type"] == "route":
                layer1_output = outputs[module["layers"][0]]
                layer2_output = outputs[module["layers"][1]] if len(module["layers"]) > 1 else None
                x = self.module_list[ind]((layer1_output, layer2_output))

            elif module["type"] == "shortcut":
                # Shortcut layer element-wise adds output tensors of two layers with same shape
                x = self.module_list[ind]((outputs[module["from"]], outputs[ind-1]))

            elif module["type"] == "yolo":
                x = self.module_list[ind](x)
                pred_of_all_scale.append(x)     # store the prediction in different scale, and concatenate them later
                # NOT CONCATENATE HERE!!!!

            else:
                raise Exception("The forward pass of module type {} is not implemented.".format(module['type']))

            outputs[ind] = x

        x = torch.cat(pred_of_all_scale, 1)
        return x

    def load_weights(self, weight_path):
        """load pretrained weights"""
        weights_path = download_pretrained_weights(weight_path)
        with open(weights_path, "rb") as f:
            # The first 5 values are header information
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        for i in range(len(self.module_list)):
            # 2nd in config corresponds to 1st in module list since net parameter is not stored in module list
            module_type = self.config[i + 1]["type"]

            # no weights in route / shortcut / detection (yolo) layers, only conv layers have weights
            if module_type == "convolutional":
                module = self.module_list[i]
                conv = module[0]
                if "batch_normalize" in self.config[i + 1]:
                    bn = module[1]
                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()
                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)




