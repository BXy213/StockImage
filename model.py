import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

TRUE_DATA_CNN_INPLANES = 64
LAYERNUM_DICT = {5: 2, 20: 3}
SETTING_DICT = {
    5: ([(5, 3)] * 10, [(1, 1)] * 10, [(1, 1)] * 10, [(2, 1)] * 10),
    20: (
        [(5, 3)] * 10,
        [(3, 1)] + [(1, 1)] * 10,
        [(2, 1)] + [(1, 1)] * 10,
        [(2, 1)] * 10,
    ),
}


class ModelManager(object):
    def __init__(
        self,
        window_size,
        drop_prob=0.50,
        regression_label=None,
        model_name='defaultmodel',
    ):
        self.ws = window_size
        assert self.ws in [5, 20]
        self.layer_number = LAYERNUM_DICT[self.ws]
        self.inplanes = TRUE_DATA_CNN_INPLANES
        self.drop_prob = drop_prob

        self.filter_size_list = SETTING_DICT[self.ws][0]
        self.stride_list = SETTING_DICT[self.ws][1]
        self.dilation_list = SETTING_DICT[self.ws][2]
        self.max_pooling_list = SETTING_DICT[self.ws][3]

        self.batch_norm = True
        self.xavier = True
        self.lrelu = True
        self.bn_loc = "bn_bf_relu"
        self.conv_layer_chanls = None
        self.regression_label = regression_label
        assert self.regression_label in [None, "raw_ret", "vol_adjust_ret"]

        self.padding_list = [(int(fs[0] / 2), int(fs[1] / 2)) for fs in self.filter_size_list]
        
        self.name = model_name

        input_size_dict = {5: (32, 15), 20: (64, 60)}
        self.input_size = input_size_dict[self.ws]

    def init_model(self, device=None):
        model = CNNModel(
            self.layer_number,
            self.input_size,
            inplanes=self.inplanes,
            drop_prob=self.drop_prob,
            filter_size_list=self.filter_size_list,
            stride_list=self.stride_list,
            padding_list=self.padding_list,
            dilation_list=self.dilation_list,
            max_pooling_list=self.max_pooling_list,
            batch_norm=self.batch_norm,
            xavier=self.xavier,
            lrelu=self.lrelu,
            bn_loc=self.bn_loc,
            conv_layer_chanls=self.conv_layer_chanls,
            regression_label=self.regression_label,
        )

        # if self.ws == 20:
        #     model = Net()

        if device is not None:
            model.to(device)

        return model

    def model_summary(self, output_path=None):
        img_size_dict = {5: (1, 32, 15), 20: (1, 64, 60)}
        device = torch.device(
            "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
        )
        model = self.init_model()
        model.to(device)
        
        if output_path is not None:
            with open(output_path, 'w', encoding='utf-8') as f:
                print(f"model name: {self.name}\n", file=f)
                print(model, file=f)
        print(self.name)
        print(model)
        summary(model, img_size_dict[self.ws])


def init_weights(m):
    if type(m) in [nn.Conv2d, nn.Conv1d]:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], np.prod(x.shape[1:]))


class CNNModel(nn.Module):
    def __init__(
        self,
        layer_number,
        input_size,
        inplanes=TRUE_DATA_CNN_INPLANES,
        drop_prob=0.50,
        filter_size_list=[(3, 3)],
        stride_list=[(1, 1)],
        padding_list=[(1, 1)],
        dilation_list=[(1, 1)],
        max_pooling_list=[(2, 2)],
        batch_norm=True,
        xavier=True,
        lrelu=True,
        conv_layer_chanls=None,
        bn_loc="bn_bf_relu",
        regression_label=None,
    ):

        self.layer_number = layer_number
        self.input_size = input_size
        self.inplanes = inplanes
        self.drop_prob = drop_prob
        self.filter_size_list = filter_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.max_pooling_list = max_pooling_list
        self.batch_norm = batch_norm
        self.xavier = xavier
        self.lrelu = lrelu
        self.conv_layer_chanls = conv_layer_chanls
        self.bn_loc = bn_loc
        
        super(CNNModel, self).__init__()
        self.conv_layers = self._init_conv_layers()

        fc_size = self._get_conv_layers_flatten_size()
        if regression_label is not None:
            self.fc = nn.Linear(fc_size, 1)
        else:
            self.fc = nn.Linear(fc_size, 2)
        if xavier:
            self.conv_layers.apply(init_weights)
            self.fc.apply(init_weights)

    def _init_conv_layers(self):
        if self.conv_layer_chanls is None:
            conv_layer_chanls = [self.inplanes * (2**i) for i in range(self.layer_number)]
        else:
            assert len(self.conv_layer_chanls) == self.layer_number
            conv_layer_chanls = self.conv_layer_chanls
        layers = []
        prev_chanl = 1
        for i, conv_chanl in enumerate(conv_layer_chanls):
            layers.append(
                self.conv_layer(
                    prev_chanl,
                    conv_chanl,
                    filter_size=self.filter_size_list[i],
                    stride=self.stride_list[i],
                    padding=self.padding_list[i],
                    dilation=self.dilation_list[i],
                    max_pooling=self.max_pooling_list[i]
                )
            )
            prev_chanl = conv_chanl
        layers.append(Flatten())
        layers.append(nn.Dropout(p=self.drop_prob))
        return nn.Sequential(*layers)

    def conv_layer(
        self,
        in_chanl: int,
        out_chanl: int,
        filter_size=(3, 3),
        stride=(1, 1),
        padding=1,
        dilation=1,
        max_pooling=(2, 2),
    ):
        assert self.bn_loc in ["bn_bf_relu", "bn_af_relu", "bn_af_mp"]

        if not self.batch_norm:
            conv = [
                nn.Conv2d(
                    in_chanl,
                    out_chanl,
                    filter_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.LeakyReLU() if self.lrelu else nn.ReLU(),
            ]
        else:
            if self.bn_loc == "bn_bf_relu":
                conv = [
                    nn.Conv2d(
                        in_chanl,
                        out_chanl,
                        filter_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(out_chanl),
                    nn.LeakyReLU() if self.lrelu else nn.ReLU(),
                ]
            elif self.bn_loc == "bn_af_relu":
                conv = [
                    nn.Conv2d(
                        in_chanl,
                        out_chanl,
                        filter_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.LeakyReLU() if self.lrelu else nn.ReLU(),
                    nn.BatchNorm2d(out_chanl),
                ]
            else:
                conv = [
                    nn.Conv2d(
                        in_chanl,
                        out_chanl,
                        filter_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.LeakyReLU() if self.lrelu else nn.ReLU(),
                ]

        layers = conv

        if max_pooling != (1, 1):
            layers.append(nn.MaxPool2d(max_pooling, ceil_mode=True))

        if self.batch_norm and self.bn_loc == "bn_af_mp":
            layers.append(nn.BatchNorm2d(out_chanl))

        return nn.Sequential(*layers)

    def _get_conv_layers_flatten_size(self):
        dummy_input = torch.rand((1, 1, self.input_size[0], self.input_size[1]))
        x = self.conv_layers(dummy_input)
        return x.shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x
