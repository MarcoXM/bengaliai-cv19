import models
import pretrainedmodels

MODEL_DISPATCHER = {
    "resnet152":models.ResNet152,
    "resnet34":models.ResNet34, # 128
    'se_net':models.SResnet, # 128
    'pnasnet':models.Pnasnet, # 128
    'inresnet':models.Inresnet, # 128
    'polynet':models.PolyNet, # 128
    'senet':models.SeNet, # 128
    'icnetvf':models.IcNetv4, # 128
}

if __name__ =="__main__":
    model = MODEL_DISPATCHER["resnet152"](pretrain=False)
    print(model)
    