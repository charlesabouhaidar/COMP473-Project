import torch
'''
As in the CycleGAN paper this generator is based on the work of Johnson et al. ()
for perceptual loss functions
'''
class ResidualBlock(torch.nn.Module):
	def __init__(self, in_features):
		super(ResidualBlock, self).__init__()

		conv_block = [  torch.nn.ReflectionPad2d(1),
						torch.nn.Conv2d(in_features, in_features, 3),
						torch.nn.InstanceNorm2d(in_features),
						torch.nn.ReLU(inplace=True),
						torch.nn.ReflectionPad2d(1),
						torch.nn.Conv2d(in_features, in_features, 3),
						torch.nn.InstanceNorm2d(in_features)  ]

		self.conv_block = torch.nn.Sequential(*conv_block)

	def forward(self, x):
		return x + self.conv_block(x)

class PerceptualGenerator(torch.nn.Module):
	def __init__(self, input_channels, output_channels, residual_blocks=3):
		super(PerceptualGenerator, self).__init__()

		model = [torch.nn.ReflectionPad2d(3),
				torch.nn.Conv2d(input_channels, 64, 7),
				torch.nn.InstanceNorm2d(64),
				torch.nn.ReLU(inplace=True)]

		#Downsampling layers
		in_features = 32
		out_features = in_features*2
		for _ in range(2):
			model += [torch.nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
					torch.nn.InstanceNorm2d(out_features),
					torch.nn.ReLU(inplace=True)]
			in_features =  out_features
			out_features = in_features*2

		#Residual Blocks
		for _ in range(residual_blocks):
			model += [ResidualBlock(in_features)]

		#Upsampling layers
		out_features = in_features//2
		for _ in range(2):
			model += [ torch.nn.ConvTranspose2d(in_features,out_features, 3, stride=2, padding=1, output_padding=1),
				torch.nn.InstanceNorm2d(out_features),
				torch.nn.ReLU(inplace=True)]
			in_features = out_features
			out_features = in_features//2

		#Output layer
		model += [torch.nn.ReflectionPad2d(3),
			torch.nn.Conv2d(64, output_channels, 7),
			torch.nn.Tanh()]

		self.model = torch.nn.Sequential(*model)

	def forward(self, x):
		return self.model(x)