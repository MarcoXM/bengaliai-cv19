import models
import pretrainedmodels

MODEL_DISPATCHER = {
    "resnet152":models.ResNet152,
    "resnet34":models.ResNet34,
    'se_net':models.SResnet,
    'pnasnet':models.Pnasnet,
    'inresnet':models.Inresnet

}

if __name__ =="__main__":
    model = MODEL_DISPATCHER["resnet152"](pretrain=False)
    print(model)
    