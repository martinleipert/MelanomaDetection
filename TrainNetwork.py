import csv
import os
import random
from FocalLoss import FocalLoss
from MelanomaDataSet import MelanomaDataset
import torch
from torch import optim, nn
from torchvision import models

PROJECT_ROOT = "/home/martin/Melanoma/"

BATCH_SIZE = 16
BATCH_SIZE_VALIDATION = 32

BREAK_ITERATIONS = 1250
LEARNING_RATE = 1e-3
LR_STEP_SIZE = 250
LR_GAMMA = 0.3


def main():

	data_path = os.path.join(PROJECT_ROOT, "Data", "images")
	csv_path = os.path.join(PROJECT_ROOT, "Data", "HAM10000_metadata.csv")

	with open(csv_path) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		csv_data = list(readCSV)

	keys = csv_data[0]

	csv_data = csv_data[1:]
	random.shuffle(csv_data)

	val_data = csv_data[0:1000]
	train_data = csv_data[1000:]

	train_data_set = MelanomaDataset(train_data, data_path)
	val_data_set = MelanomaDataset(val_data, data_path)

	training_loader = torch.utils.data.DataLoader(train_data_set, batch_size=BATCH_SIZE)
	validation_loader = torch.utils.data.DataLoader(val_data_set, batch_size=BATCH_SIZE_VALIDATION)

	# Evaluate on CUDA if possible
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_network = models.resnet18(pretrained=True)
	model_network.fc = nn.Sequential(nn.Linear(512, BATCH_SIZE), nn.ReLU(), nn.Linear(BATCH_SIZE, 7), nn.LogSoftmax(dim=1))

	model_network.to(device)

	training_criterion = FocalLoss(gamma=2, size_average=True).to(device)
	validation_criterion = FocalLoss(gamma=2, size_average=True).to(device)


	optimizer = optim.Adam(model_network.parameters(), lr=LEARNING_RATE)
	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

	iteration_count = 0
	epoch = 0
	running_loss = 0

	model_network.train()

	while True:
		epoch += 1
		# Training
		for inputs, labels in training_loader:
			iteration_count += 1
			exp_lr_scheduler.step()

			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			logps = model_network(inputs)
			loss = training_criterion(logps, labels)
			loss.backward()
			optimizer.step()

			current_loss = loss.detach().item()
			running_loss += current_loss

			print("Iteration %i - Loss %f" % (iteration_count, current_loss))

			if (iteration_count % BREAK_ITERATIONS) == 0:
				break
		if (iteration_count % BREAK_ITERATIONS) == 0:
			break


if __name__ == "__main__":
	main()
