import models
import pretrainedmodels

MODEL_DISPATCHER = {
    "resnet101":models.ResNet101, # memory 
    "resnet34":models.ResNet34, # 128 done finish weight 5,3,2
    'se_net':models.SResnet, #224x224 128 # weights 7,2,1
    'pnasnet':models.Pnasnet, # 331x 331 erdos 2  # weights 7,2,1 
    'inresnet':models.Inresnet, #299x 299  64 can   # weight 7,2,1
    'polynet':models.PolyNet, # 128 # weight 7,2,1
    'senet':models.SeNet, # 128 # weights 7,2,1
    'icnetvf':models.IcNetv4, # 128 weights 7,2,1
    'effinet':models.Effinet,
    'gnet':models.Gnet,
}

if __name__ =="__main__":
    model = MODEL_DISPATCHER["resnet152"](pretrain=False)
    print(model)
    