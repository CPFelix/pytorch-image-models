import timm
from pprint import pprint
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch

# v0.1-rsb-weights
# TEST 8
if __name__ == "__main__":
    # print models
    model_names = timm.list_models(pretrained=True)
    # model_names = timm.list_models('*ran*')
    pprint(model_names)

    # model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # model.eval()
    #
    # config = resolve_data_config({}, model=model)
    # transform = create_transform(**config)
    #
    # # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    # # urllib.request.urlretrieve(url, filename)
    #
    # filename = "./cat.jpg"
    # img = Image.open(filename).convert('RGB')
    # tensor = transform(img).unsqueeze(0)  # transform and add batch dimension
    #
    # with torch.no_grad():
    #     out = model(tensor)
    # probabilities = torch.nn.functional.softmax(out[0], dim=0)
    # print(probabilities.shape)
    #
    # # Get imagenet class mappings
    # # url, filename = (
    # # "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    # # urllib.request.urlretrieve(url, filename)
    #
    # with open("./imagenet_classes.txt", "r") as f:
    #     categories = [s.strip() for s in f.readlines()]
    #
    # # Print top categories per image
    # top5_prob, top5_catid = torch.topk(probabilities, 5)
    # for i in range(top5_prob.size(0)):
    #     print(categories[top5_catid[i]], top5_prob[i].item())