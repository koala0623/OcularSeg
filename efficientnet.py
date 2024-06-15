import timm

all_densenet_models = timm.list_models('*efficientnet*', pretrained = True)
print(all_densenet_models)