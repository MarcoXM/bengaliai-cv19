import models
import pretrainedmodels

MODEL_DISPATCHER = {
    "resnet152":models.ResNet152

}

if __name__ =="__main__":
    model = MODEL_DISPATCHER["resnet152"](pretrain=False)
    print(model)