import torch

'''
Basic CNN
'''
class CNNDiscriminator(torch.nn.Module):
	def __init__(self, input_channels):
		super(CNNDiscriminator, self).__init__()

		# A bunch of convolutions one after another
		model = [   torch.nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
					torch.nn.ReLU(inplace=True) ]

		model += [  torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),
					torch.nn.InstanceNorm2d(128), 
					torch.nn.ReLU(inplace=True) ]

		model += [  torch.nn.Conv2d(128, 128, 4, stride=2, padding=1),
					torch.nn.InstanceNorm2d(256), 
					torch.nn.ReLU(inplace=True) ]

		model += [  torch.nn.Conv2d(128, 256, 4, padding=1),
					torch.nn.InstanceNorm2d(512), 
					torch.nn.ReLU(inplace=True) ]

		# Fully connected layer
		model += [torch.nn.Conv2d(256, 1, 4, padding=1)]

		self.model = torch.nn.Sequential(*model)

	def forward(self, x):
		x =  self.model(x)
		# Average pooling and flatten
		return torch.nn.functional.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)