import torch
import torch.nn as nn
import torch.nn.functional as F

### Components
def conv(with_batch_norm, input_channel, output_channel, kernel_size=3, stride=1, padding=0, activate=True):
    if with_batch_norm & activate:
        layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    elif ~with_batch_norm & activate:
        layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    elif with_batch_norm & ~activate:
        layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channel)
        )
    else:
        raise(Exception('No need to use compact layers.'))
    return layer

class PosELU(nn.Module):
    '''
    Description:
        A positive exponential linear unit/layer.
        Only the negative part of the exponential retains.
        The positive part is linear: y=x+1.
    '''
    def __init__(self, offset=0) -> None:
         super().__init__()
         self.offset = offset

    def forward(self, x):
        l = nn.ELU() # ELU: max(0,x)+min(0,α∗(exp(x)−1))
        return torch.add(l(x), 1+self.offset) # assure no negative sigma produces!!!

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, with_batch_norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = conv(with_batch_norm, in_channels,  mid_channels, padding=1)
        self.conv2 = conv(with_batch_norm, mid_channels, out_channels, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, with_batch_norm=True):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, with_batch_norm=with_batch_norm))

    def forward(self, x):
        return self.down_conv(x)

class UpBlock(nn.Module): # with skip connection
    def __init__(self, in_channels, out_channels, doubleconv=True, with_batch_norm=True, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if doubleconv:
                self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels//2, with_batch_norm=with_batch_norm)
            else:
                self.conv = conv(in_channels, out_channels, with_batch_norm=with_batch_norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            if doubleconv:
                self.conv = DoubleConv(in_channels, out_channels, with_batch_norm=with_batch_norm)
            else:
                self.conv = conv(in_channels, out_channels, with_batch_norm=with_batch_norm)

    def forward(self, x1, x2):
        # x1 is the front feature map, x2 is the skip-connection feature map
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

### Backbones
class UNetTypeEncoder(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, channels=(64,128,256,512,512), with_batch_norm=True):
        super(UNetTypeEncoder,self).__init__()
        chs = channels

        self.inc = DoubleConv(in_channels, chs[0], with_batch_norm=with_batch_norm)
        self.downs = nn.ModuleList()
        for i in range(len(chs)-1):
            self.downs.append(DownBlock(chs[i], chs[i+1], with_batch_norm=with_batch_norm))
        # self.out = nn.Conv2d(512, num_classes, kernel_size=1)
        self.out = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x) -> list:
        features = []
        x = self.inc(x)
        features.append(x) # full resolution feature
        for down in self.downs:
            x = down(x)
            features.append(x)
        x = self.out(x)
        features.append(x) # final feature
        return features

class UNetTypeDecoder(nn.Module):
    # batch x channel x height x width
    def __init__(self, encoder_channels, decoder_channels, out_channel=1, with_batch_norm=True):
        super(UNetTypeDecoder,self).__init__()

        self.inc = DoubleConv(encoder_channels[-1], out_channels=encoder_channels[-1], with_batch_norm=with_batch_norm)

        up_in_chs  = [encoder_channels[-1]] + decoder_channels[:-1]
        up_out_chs = up_in_chs # for bilinear
        dec_in_chs  = [enc + dec for enc, dec in zip(encoder_channels[::-1], up_out_chs)] # add feature channels
        dec_out_chs = decoder_channels
        self.decoder = nn.ModuleList()
        for in_chs, out_chs in zip(dec_in_chs, dec_out_chs):
            self.decoder.append(UpBlock(in_chs, out_chs, bilinear=True, with_batch_norm=with_batch_norm))
        
        self.out = nn.Conv2d(decoder_channels[-1], out_channel, kernel_size=1)

    def forward(self, features):
        features = features[::-1]
        x = self.inc(features[0])
        for feature, dec in zip(features[1:], self.decoder):
            x = dec(x, feature)
        logits = self.out(x)
        return logits

class UNet(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, num_classes=1, with_batch_norm=True, bilinear=True, lite:bool=True):
        super(UNet,self).__init__()

        if lite:
            chs = [16, 32,  64,  128, 256]
        else:
            chs = [64, 128, 256, 512, 1024]

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(in_channels, chs[0], with_batch_norm=with_batch_norm)
        self.down1 = DownBlock(chs[0], chs[1], with_batch_norm=with_batch_norm)
        self.down2 = DownBlock(chs[1], chs[2], with_batch_norm=with_batch_norm)
        self.down3 = DownBlock(chs[2], chs[3], with_batch_norm=with_batch_norm)
        self.down4 = DownBlock(chs[3], chs[4]//factor, with_batch_norm=with_batch_norm)
        self.up1 = UpBlock(chs[4], chs[3]//factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up2 = UpBlock(chs[3], chs[2]//factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up3 = UpBlock(chs[2], chs[1]//factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up4 = UpBlock(chs[1], chs[0], bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.outc = nn.Conv2d(chs[0], num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x0 = self.up1(x5, x4)
        x0 = self.up2(x0, x3)
        x0 = self.up3(x0, x2)
        x0 = self.up4(x0, x1)
        logits = self.outc(x0)
        return logits

### References
class YNetEncoder(nn.Module):
	def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
		"""
		Encoder model
		:param in_channels: int, semantic_classes + obs_len
		:param channels: list, hidden layer channels
		"""
		super(YNetEncoder, self).__init__()
		self.stages = nn.ModuleList()

		# First block
		self.stages.append(nn.Sequential(
			nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
		))

		# Subsequent blocks, each starting with MaxPool
		for i in range(len(channels)-1):
			self.stages.append(nn.Sequential(
				nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
				nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True),
				nn.Conv2d(channels[i+1], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
				nn.ReLU(inplace=True)))

		# Last MaxPool layer before passing the features into decoder
		self.stages.append(nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))

	def forward(self, x):
		# Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
		features = []
		for stage in self.stages:
			x = stage(x)
			features.append(x)
		return features

class YNetDecoder(nn.Module):
	def __init__(self, encoder_channels, decoder_channels, output_len, extra_channels:int=None):
		"""
		Decoder models
		:param encoder_channels: list, encoder channels, used for skip connections
		:param decoder_channels: list, decoder channels
		:param output_len: int, pred_len
		:param num_waypoints: None or int, if None -> Goal and waypoint predictor, if int -> number of waypoints
		"""
		super(YNetDecoder, self).__init__()

		# The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
		if extra_channels:
			encoder_channels = [channel+extra_channels for channel in encoder_channels]
		encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
		center_channels = encoder_channels[0]

		# The center layer (the layer with the smallest feature map size)
		self.center = nn.Sequential(
			nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True)
		)

		# Determine the upsample channel dimensions
		upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
		upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

		# Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
		self.upsample_conv = [
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)]
		self.upsample_conv = nn.ModuleList(self.upsample_conv)

		# Determine the input and output channel dimensions of each layer in the decoder
		# As we concat the encoded feature and decoded features we have to sum both dims
		in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
		out_channels = decoder_channels

		self.decoder = [nn.Sequential(
			nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.ReLU(inplace=True))
			for in_channels_, out_channels_ in zip(in_channels, out_channels)]
		self.decoder = nn.ModuleList(self.decoder)

		# Final 1x1 Conv prediction to get our heatmap logits (before softmax)
		self.predictor = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

	def forward(self, features):
		# Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
		features = features[::-1]  # reverse the order of encoded features, as the decoder starts from the smallest image
		center_feature = features[0]
		x = self.center(center_feature)
		for i, (feature, module, upsample_conv) in enumerate(zip(features[1:], self.decoder, self.upsample_conv)):
			x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # bilinear interpolation for upsampling
			x = upsample_conv(x)  # 3x3 conv for upsampling
			x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
			x = module(x)  # Conv
		x = self.predictor(x)  # last predictor layer
		return x
