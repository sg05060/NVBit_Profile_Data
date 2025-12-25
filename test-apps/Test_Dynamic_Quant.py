import time
import argparse
import onnxruntime as ort
from tqdm import tqdm
from torchvision import transforms
from transformers import BertTokenizerFast
import os
import torch
from PIL import Image
import torchvision as tv
import glob
import onnx
import argparse

parser = argparse.ArgumentParser(description="Image classification using Quant Model")

# Add an argument for the batch size
parser.add_argument("--model_path", type=str, help="model path")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
parser.add_argument("--iter_max", type=int, default=4, help="iter size for inference")
args = parser.parse_args()

model_path = args.model_path
batch_size = args.batch_size
iter_max = args.iter_max

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#################Sysnet to Target###################
synset_to_target = {}
f = open("./synset_words.txt", "r")
index = 0
for line in f:
    parts = line.split(" ")
    synset_to_target[parts[0]] = index
    index = index + 1
f.close()
####################################################

#################Make Dataset##########################################################
preprocess = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(244),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def tar_transform(synset):
  return synset_to_target[synset]

class ImageNetValDataset(torch.utils.data.Dataset):
  def __init__(self, img_dir, transform=None, target_transform=None):
      self.img_dir = img_dir
      self.img_paths = sorted(glob.glob(img_dir + "*/*.JPEG"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
      self.transform = transform
      self.target_transform = target_transform

  def __len__(self):
      return len(self.img_paths)

  def __getitem__(self, idx):
      img_path = self.img_paths[idx]
      image = Image.open(img_path).convert('RGB')
      synset = img_path.split("/")[-2]
      label = synset
      if self.transform:
          image = self.transform(image)
      if self.target_transform:
          label = self.target_transform(label)
      return image, label

ds = ImageNetValDataset("/media/3/sg05060/UVMC/0_dataset/val/", transform=preprocess, target_transform=tar_transform)
offset = 500
ds = torch.utils.data.Subset(ds, list(range(offset)))

dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
########################################################################################

################################ Run Test ################################
ort_sess = ort.InferenceSession(model_path,providers=['CUDAExecutionProvider'])

iter = 0
for img_batch, label_batch in dl:

  ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img_batch)}
  ort_outs = ort_sess.run(None, ort_inputs)[0]

  #ort_preds = np.argmax(ort_outs, axis=1)
  #correct_onnx += np.sum(np.equal(ort_preds, to_numpy(label_batch)))

  #if torch.cuda.is_available():
  #  img_batch = img_batch.to('cuda')
  #  label_batch = label_batch.to('cuda')

  #with torch.no_grad():
  #  pt_outs = model_pt(img_batch)

  #pt_preds = torch.argmax(pt_outs, dim=1)
  #correct_pt += torch.sum(pt_preds == label_batch)

  #tot_abs_error += np.sum(np.abs(to_numpy(pt_outs) - ort_outs))
  iter += 1
  if iter == iter_max :
        break

############################################################################


