from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class MelanomaDataset(Dataset):
	# {'nv': 6705, 'mel': 1113, 'bkl': 1099, 'df': 115, 'akiec': 327, 'vasc': 142, 'bcc': 514}

	__TRANSLATOR =  {
		"bkl" : 0,
		"df" : 1,
		"nv" : 2,
		"mel" : 3,
		"vasc" : 4,
		"bcc" : 5,
		"akiec" : 6
	}

	def __init__(self, data_list, directory_root):
		self.__data_list = data_list
		self.__directory_root = directory_root

		pass

	@classmethod
	def load_img(cls, im_path):
		image = Image.open(im_path)
		image.load()
		image = image.convert('RGB')
		to_tensor = transforms.ToTensor()

		image = to_tensor(image)
		return image

	def __getitem__(self, index):
		lesion_id, image_id, dx, dx_type, age, sex, localization  = self.__data_list[index]

		im_path = os.path.join(self.__directory_root, "%s.jpg" % image_id)

		img = MelanomaDataset.load_img(im_path)

		target = MelanomaDataset.__TRANSLATOR[dx]

		return img, target

	def __len__(self):
		return len(self.__data_list)
