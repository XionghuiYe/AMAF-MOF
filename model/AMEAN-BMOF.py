from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import repeat, rearrange
from torchinfo import summary
import copy
from common import DeformConv2d, my_DeformConv2d
from LGI_Former import LGBlock

class RMSA(nn.Module):
    def __init__(self, dim=16, num_heads=4, bias=False):
        super(RMSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.deform_conv = my_DeformConv2d(dim, dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, clone, x, u, v):
        clone = self.deform_conv(clone, u, v)
        clone = torch.flatten(clone, 2)
        # q = rearrange(q, 'b (head c) s -> b head c s', head=self.num_heads)
        # k = v = rearrange(x, 'b (head c) s -> b head c s', head=self.num_heads)
        k = v = rearrange(clone, 'b (head c) s -> b head c s', head=self.num_heads)
        q = rearrange(x, 'b (head c) s -> b head c s', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c s -> b (head c) s', head=self.num_heads)

        return out




# Define MEAN_Spot model
class torch_AMEAN_Spot(nn.Module):
    def __init__(self, depth=12, drop_path_rate=0.):
        super(torch_AMEAN_Spot, self).__init__()
        # channel 1
        self.channel_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # channel 2
        self.channel_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=5, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # channel 3
        self.channel_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )

        self.lg_region_tokens = nn.Parameter(torch.zeros(1, 196))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            LGBlock(dim=196, num_heads=4, drop_path=dpr[i])
            for i in range(depth)])
        # interpretation
        self.interpretation = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=3332, out_features=1666),
            nn.ReLU(),
            nn.Linear(in_features=1666, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=1),
        )

    def forward(self, x1, x2, x3):
        inputs_1 = x1
        inputs_2 = x2
        inputs_3 = x3
        # channel 1
        channel_1 = self.channel_1(inputs_1)
        # channel 2
        channel_2 = self.channel_2(inputs_2)
        # channel 3
        channel_3 = self.channel_3(inputs_3)
        # merge
        merged = torch.cat((channel_1, channel_2, channel_3), 1)
        x = torch.flatten(merged,2)
        b = x.size(0)
        region_tokens = repeat(self.lg_region_tokens, 'n c -> b n c', b=b)
        x = torch.cat([region_tokens, x], dim=1)  # (b, nt*nh*nw, 1+thw, c)
        for blk in self.blocks:
            x = blk(x)
        # x = x[:, 1:]
        # x = x[:, 0]
        # interpretation
        outputs = self.interpretation(x)
        return outputs

class torch_p_AMEAN_Spot(nn.Module):
    def __init__(self, depth=12, drop_path_rate=0.):
        super(torch_p_AMEAN_Spot, self).__init__()
        # channel 1
        self.channel_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # channel 2
        self.channel_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=5, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # channel 3
        self.channel_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # # interpretation
        # self.interpretation = nn.Sequential(
        #     # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        #     nn.Flatten(),
        #     nn.LazyLinear(out_features=1666),
        #     nn.ReLU(),
        #     nn.LazyLinear(out_features=400),
        #     nn.ReLU(),
        #     nn.LazyLinear(out_features=1),
        # )
        self.lg_region_tokens = nn.Parameter(torch.zeros(1, 196))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            LGBlock(dim=196, num_heads=4, drop_path=dpr[i])
            for i in range(depth)])

    def forward(self, x1, x2, x3):
        inputs_1 = x1
        inputs_2 = x2
        inputs_3 = x3
        # channel 1
        channel_1 = self.channel_1(inputs_1)
        # channel 2
        channel_2 = self.channel_2(inputs_2)
        # channel 3
        channel_3 = self.channel_3(inputs_3)
        # merge
        merged = torch.cat((channel_1, channel_2, channel_3), 1)
        clone = merged.clone()
        x = torch.flatten(merged,2)
        b = x.size(0)
        region_tokens = repeat(self.lg_region_tokens, 'n c -> b n c', b=b)
        x = torch.cat([region_tokens, x], dim=1)  # (b, nt*nh*nw, 1+thw, c)
        for blk in self.blocks:
            x = blk(x)
        # outputs = self.interpretation(x)
        return x, clone


# Define MEAN_Recog_TL model
class torch_AMEAN_Recog_TL(nn.Module):
    def __init__(self, p_model_spot):
        super(torch_AMEAN_Recog_TL, self).__init__()
        for param in p_model_spot.parameters():
            param.requires_grad = False
        self.model_p = p_model_spot
        self.recog_attn = RMSA(dim=16, num_heads=4)
        # interpretation
        self.interpretation = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=1568),
            nn.ReLU(),
            nn.Linear(in_features=1568, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=3),
        )

    def forward(self, x1, x2, x3):
        u = x1.clone()
        v = x2.clone()
        x, clone = self.model_p(x1, x2, x3)
        x = x[:, 1:]
        x = self.recog_attn(clone, x, u, v)
        outputs = self.interpretation(x)
        return outputs


class torch_q_model_spot(nn.Module):
    def __init__(self):
        super(torch_q_model_spot, self).__init__()
        # interpretation
        self.interpretation = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=3332, out_features=1666),
            nn.ReLU(),
            nn.Linear(in_features=1666, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=1),
        )

    def forward(self, x):
        outputs = self.interpretation(x)
        return outputs


class torch_q_model_recog(nn.Module):
    def __init__(self):
        super(torch_q_model_recog, self).__init__()
        self.recog_attn = RMSA(dim=16, num_heads=4)
        # interpretation
        self.interpretation = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=1568),
            nn.ReLU(),
            nn.Linear(in_features=1568, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=3),
        )

    def forward(self, x, clone, u, v):
        x = x[:, 1:]
        x = self.recog_attn(clone, x, u, v)
        outputs = self.interpretation(x)
        return outputs


# Define MEAN_Spot_Recog_TL model
class torch_AMEAN_Spot_Recog_TL(nn.Module):
    def __init__(self, torch_p_AMEAN_Spot, q_model_spot, q_model_recog):
        super(torch_AMEAN_Spot_Recog_TL, self).__init__()
        self.model_p = torch_p_AMEAN_Spot
        self.q_model_spot = q_model_spot
        self.q_model_recog = q_model_recog
    def forward(self, x1, x2, x3):
        u = x1.clone()
        v = x2.clone()
        w = x3.clone()
        base_output, clone = self.model_p(x1, x2, x3)
        spot_output = self.q_model_spot(base_output)
        recog_output = self.q_model_recog(base_output, clone, u, v)
        return spot_output, recog_output


def load_s_r_dict(model_spot, model_recog):
    order_dict = OrderedDict()
    kmodel_params = model_spot.state_dict()
    #
    key_list = list(kmodel_params.keys())
    #
    for key in key_list[:-6]:
        new_key = "model_p" + "." + key
        order_dict[new_key] = kmodel_params[key]
    #
    for key in key_list[-6:]:
        new_key = "q_model_spot" + "." + key
        order_dict[new_key] = kmodel_params[key]
    # #
    # # # for key in list(kmodel_params.keys()):
    # # #     if "model_p" not in key and "q_model_spot" not in key:
    # # #         del kmodel_params[key]
    # #
    # # for name, param in order_dict.items():
    # #     print(name, param.shape)
    # #
    recog_model_static = model_recog.state_dict()
    recog_list = list()
    for key in recog_model_static:
        recog_list.append(key)
    #
    for key in recog_list[-16:]:
        new_key = "q_model_recog" + "." + key
        order_dict[new_key] = recog_model_static[key]

    return order_dict

# # Example usage:
# # Create instances of models
# model_spot = torch_MEAN_Spot()
# model_recog = MEAN_Recog_TL(model_spot, num_classes=10)
# model_spot_recog = MEAN_Spot_Recog_TL(model_spot, model_recog)
#
# # Define optimizer and loss function
# optimizer = optim.Adam(model_spot_recog.parameters(), lr=0.001)
# criterion_spot = nn.MSELoss()
# criterion_recog = nn.CrossEntropyLoss()
#
# # Training loop and so on...

if __name__ == '__main__':
    model_spot = torch_AMEAN_Spot().to('cuda')

    p_model_spot = torch_p_AMEAN_Spot().to('cuda')
    q_model_spot = torch_q_model_spot().to('cuda')
    q_model_recog = torch_q_model_recog().to('cuda')
    summary(model_spot, [(2, 1, 42, 42), (2, 1, 42, 42), (2, 1, 42, 42)], device='cuda')
    model_static = model_spot.state_dict()
    for name, param in model_static.items():
        print(name, param.shape)
    # # print("----------------")
    # # # print(model_static["conv1_1.weight"])
    # # #
    print("----------------")
    print("----------------")
    model_static_p = copy.deepcopy(p_model_spot.state_dict())
    for name, param in model_static_p.items():
        print(name, param.shape)
    # # # print(model_static_p["conv1_1.weight"])
    # # #
    model_static.pop("interpretation.1.weight")
    model_static.pop("interpretation.1.bias")
    model_static.pop("interpretation.3.weight")
    model_static.pop("interpretation.3.bias")
    model_static.pop("interpretation.5.weight")
    model_static.pop("interpretation.5.bias")
    # # # print(model_spot)
    p_model_spot.load_state_dict(model_static, strict=True)
    print("load ok!")
    # # # model_static_pp = p_model_spot.state_dict()
    # # # for name, param in model_static_pp.items():
    # # #     print(name, param.shape)
    # # # print(model_static_pp["conv1_1.weight"])
    # # #
    print("----------------")
    model_recog = torch_AMEAN_Recog_TL(p_model_spot).to('cuda')
    summary(model_recog, [(2, 1, 42, 42), (2, 1, 42, 42), (2, 1, 42, 42)], device='cuda')
    # #
    MSRT = torch_AMEAN_Spot_Recog_TL(p_model_spot, q_model_spot, q_model_recog)
    summary(MSRT, [(2, 1, 42, 42), (2, 1, 42, 42), (2, 1, 42, 42)], device='cuda')
    model_static = MSRT.state_dict()
    for name, param in model_static.items():
        print(name, param.shape)
    #
    print("--------model_spot--------")
    model_spot_static = model_spot.state_dict()
    for name, param in model_spot_static.items():
        print(name, param.shape)
    # #
    print("--------model_recog--------")
    model_recog_static = model_recog.state_dict()
    for name, param in model_recog_static.items():
        print(name, param.shape)
    #
    print("--------spot_model_new_static--------")
    order_dict = OrderedDict()
    kmodel_params = model_spot.state_dict()
    #
    key_list = list(kmodel_params.keys())
    #
    for key in key_list[:-6]:
        new_key = "model_p" + "." + key
        order_dict[new_key] = kmodel_params[key]
    # #
    for key in key_list[-6:]:
        new_key = "q_model_spot" + "." + key
        order_dict[new_key] = kmodel_params[key]
    # # #
    # # # # for key in list(kmodel_params.keys()):
    # # # #     if "model_p" not in key and "q_model_spot" not in key:
    # # # #         del kmodel_params[key]
    # # #
    # # # for name, param in order_dict.items():
    # # #     print(name, param.shape)
    # # #
    print("--------recog_model_new_static--------")
    recog_model_static = model_recog.state_dict()
    recog_list = list()
    for key in recog_model_static:
        recog_list.append(key)
    #
    for key in recog_list[-16:]:
        new_key = "q_model_recog" + "." + key
        order_dict[new_key] = recog_model_static[key]
    # #
    missing_keys, unexpected_keys = MSRT.load_state_dict(order_dict, strict=False)
    print(missing_keys)
    print(unexpected_keys)
    # #
    # # for name, param in order_dict.items():
    # #     print(name, param.shape)

