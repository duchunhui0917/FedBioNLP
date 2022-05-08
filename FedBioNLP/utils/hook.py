import torch
from torch.nn import Conv2d, Linear, AdaptiveAvgPool2d


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.layer1 = Linear(in_features=32, out_features=64)
        self.avgpool = AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        return x


class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        '''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
        '''
        self.fea = fea_out


def get_feas_by_hook(model):
    """
    提取Conv2d后的feature，我们需要遍历模型的module，然后找到Conv2d，把hook函数注册到这个module上；
    这就相当于告诉模型，我要在Conv2d这一层，用hook_fun处理该层输出的feature.
    由于一个模型中可能有多个Conv2d，所以我们要用hook_feas存储下来每一个Conv2d后的feature
    """
    fea_hooks = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks[n] = cur_hook
    return fea_hooks

# model = Model()
# fea_hooks = get_feas_by_hook(model)  # 调用函数，完成注册即可
#
# for i in range(10):
#     x = torch.randn([32, 3, 224, 224])
#     out = model(x)
# print('The number of hooks is:', len(fea_hooks))
# print('The shape of the first Conv2D feature is:', fea_hooks[0].fea.shape)
