from .BMLQ_Dataset import corr
import torch
from torchvision.transforms import v2


def vizualize_preprocess(inputs, targets, output):
    return vizualize(inputs, targets, output, preprocess_for_resnet=True)

def vizualize(inputs, targets, output, preprocess_for_resnet=False):
    invTrans = v2.Compose([ v2.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                v2.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    if preprocess_for_resnet:
        inputs = invTrans(inputs)

    output_value, output = torch.max(output, dim=1)
    output = output.to(torch.uint8)#torch.max -> torch.int64 !
    output = torch.unsqueeze(output, 1)
    output = torch.cat((output, output, output), dim=1)

    targets_value, targets = torch.max(targets, dim=1)
    targets = targets.to(torch.uint8)#torch.argmax -> torch.int64 !
    targets = torch.unsqueeze(targets, 1)
    targets = torch.cat((targets, targets, targets), dim=1)
    for i in corr:
        targets[:, 0][targets[:, 0] == corr[i]["input"]] = corr[i]["output"][0]
        targets[:, 1][targets[:, 1] == corr[i]["input"]] = corr[i]["output"][1]
        targets[:, 2][targets[:, 2] == corr[i]["input"]] = corr[i]["output"][2]

        if not i == "Ignore":
            output[:, 0][output[:, 0] == corr[i]["input"]-1] = corr[i]["output"][0]
            output[:, 1][output[:, 1] == corr[i]["input"]-1] = corr[i]["output"][1]
            output[:, 2][output[:, 2] == corr[i]["input"]-1] = corr[i]["output"][2]

    output = output * torch.unsqueeze(output_value, 1)
    targets = targets * torch.unsqueeze(targets_value, 1)
    vizualisation = torch.cat((torch.round((inputs/2)*255+targets/2).to(torch.uint8), torch.round((inputs/2)*255+output/2).to(torch.uint8)), dim=3)#output.to(torch.uint8)), dim=3)
    return vizualisation[0]
