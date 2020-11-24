import torch
from models.generators import PerceptualGenerator
from models.discriminators import CNNDiscriminator

class CycleGAN():
	def __init__(self, input_channels, output_channels, res_blocks=3, cuda=False):
		self.GenA = PerceptualGenerator(input_channels, output_channels, res_blocks)
		self.GenB = PerceptualGenerator(input_channels, output_channels, res_blocks)
		self.DiscA = CNNDiscriminator(input_channels)
		self.DiscB = CNNDiscriminator(input_channels)

		if cuda:
			self.GenA.cuda()
			self.GenB.cuda()
			self.DiscA.cuda()
			self.DiscB.cuda()




