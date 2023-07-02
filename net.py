import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from pygini import gini

from transformations import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



########################################################################
# models
########################################################################

class Autoencoder(nn.Module):
    """
    the autoencoder class
    """
    def __init__(self, encoder, decoder, lambda_1, lambda_2, lambda_tv):
        super(Autoencoder, self).__init__()
        # encoder
        self.encoder = encoder

        # decoder
        self.decoder = decoder

        # loss
        self.l2_loss = L2Loss()

        # loss balancing
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_tv = lambda_tv

        self.iter=0

    def forward(self, input):
        # encode input
        input_encoded = self.encoder(input)['r41']


        # # get the amplitude and phase of the input
        input_amplitude, input_phase = fft_magnitude_phase(input_encoded)

        # save the amplitude and phase of the input to file
        np.save("./npy/input_amplitude_{}.npy".format(self.iter), input_amplitude.cpu().detach().numpy())
        np.save("./npy/input_phase_{}.npy".format(self.iter), input_phase.cpu().detach().numpy())

        self.iter+=1

        # load the amplitude and phase of the input from file
        # other_amplitude = torch.from_numpy(np.load("./npy/input_amplitude.npy")).to(device)

        #print("input_amplitude shape: {}".format(input_amplitude.shape))
        # print("input_phase shape: {}".format(input_phase.shape))

        # input_amplitude, input_phase = fft_magnitude_phase(input_encoded)
        # use ifft to reconstruct the input
        # input_encoded = fft_magnitude_phase_reconstruction(other_amplitude, input_phase)




        # fig=plt.figure(figsize=(8, 4))
        # grid = ImageGrid(fig, 111,
        #                     nrows_ncols=(1, 5),
        #                     axes_pad=0.1,
        #                     share_all=True,
        #                     cbar_location="right",
        #                     cbar_mode="single",
        #                     cbar_size="7%",
        #                     cbar_pad=0.15
        #                 )   

        # vivsuallize the feature map
        # visualize_data = input_encoded[0]
        # visualize_data = visualize_data.cpu().detach().numpy()
        # visualize_data = np.transpose(visualize_data, (1, 2, 0))
        # # only show the first channel
        # for i in range(0,1):
        #     visualize_data = visualize_data[:, :, i]
        #     # set up the subplot title
        #     grid[0].set_title("original")
        #     grid[0].imshow(visualize_data)




        # input_encoded = identity_transformation(input_encoded)
        
        # input_encoded = relu_transformation(input_encoded)

        # input_encoded = activation_mask_transformation(input_encoded)

        # input_encoded = ratio_and_bias_transformation(input_encoded)

        # input_encoded,ll = wavelet_transformation(input_encoded)

        # print(input_encoded.shape)

        # input_encoded_l,input_encoded_h = fft_freq_filter_trasformation(input_encoded)
        # input_encoded=input_encoded_h

        # print(input_encoded.shape)

        # for i in range(0,1):
        #     visualize_data2 = input_encoded_l[0][i]
        #     # set up the subplot title
        #     grid[1].set_title("low")
        #     grid[1].imshow(visualize_data2)

        # for i in range(0,1):
        #     visualize_data3 = input_encoded_h[0][i]
        #     # set up the subplot title
        #     grid[2].set_title("high")
        #     grid[2].imshow(visualize_data3)


        # set up the colorbar
        # grid.cbar_axes[0].colorbar(grid[0].images[0])
        # plt.show()

        # calculate the variance of each channel
        # variance = torch.var(input_encoded, dim=(2, 3))
        # # sort the variance
        # variance, index = torch.sort(variance, dim=1, descending=True)
        # # show the distribution of the variance
        # plt.plot(variance[0])
        # plt.show()

        # # get 5 channels evenly distributed
        # index = index[:, ::int(variance.shape[1] / 5)]
        
        # # visualize the corresponding feature map
        # for i in range(0,5):
        #     visualize_data = input_encoded[0][index[0][i]]
        #     # set the title of the subplot as the variance of the corresponding channel with 4 decimal place
        #     grid[i].set_title("variance: {:.4f}".format(variance[0][i]))
        #     grid[i].imshow(visualize_data)
        
        # set up the colorbar
        # grid.cbar_axes[0].colorbar(grid[0].images[0])
        # plt.show()



        # get output

        # visualize the feature map using matplotlib

        # set up color bar
        # print("amp:",input_amplitude[0][0])


        # calculate the gini index
        # plt.subplot(1, 3, 1)
        # plt.title("channel")
        # plt.imshow(input_encoded[0][0].cpu().detach().numpy())
        # # show the mag and phase of the input
        # plt.subplot(1, 3, 2)
        # plt.title("amplitude")
        # plt.imshow(input_amplitude[0][0].cpu().detach().numpy())
        # # plt.colorbar()
        # plt.subplot(1, 3, 3)
        # plt.title("phase")
        # plt.imshow(input_phase[0][0].cpu().detach().numpy())
        
        # plt.show()

        # # calculate the gini index of the channle value:
        # gini_index = gini(input_encoded[0][0].flatten().cpu().detach().numpy())
        # print("channel gini:",gini_index)
        # # calculate the gini index of the amplitude value:
        # gini_index = gini(input_amplitude[0][0].flatten().cpu().detach().numpy())
        # print("amplitude gini:",gini_index)
        # # calculate the gini index of the phase value:
        # gini_index = gini(input_phase[0][0].flatten().cpu().detach().numpy())
        # print("phase gini:",gini_index)
        # exit()




        output2 = self.decoder(input_encoded)
        
        output = input-output2

        

        # encode output
        output_encoded = self.encoder(output)['r41']

        # MSELoss(self, input, target) => input_features are the target
        loss, feature_loss, per_pixel_loss = self.calculate_loss(output, input, output_encoded, input_encoded)

        return output2, loss, feature_loss, per_pixel_loss

    

    def calculate_loss(self, input, target, input_features, target_features):
        """
        calculates the network loss (feature loss and per-pixel loss and TV loss)
        @param input: output image of the network
        @param target: original input image of the network
        @param input_features: encoding of image @param input
        @param target_features: encoding of image @param target
        @return:
        """
        # feature loss on relu_4
        content_feature_loss = self.l2_loss(input_features.to(device), target_features.to(device))

        # per pixel loss on the images
        per_pixel_loss = self.l2_loss(input, target).to(device)

        # tv regularizer
        tv_regularizer = self.tv_regularizer(input)

        # loss is sum of losses
        loss = self.lambda_tv * tv_regularizer + \
               self.lambda_1 * content_feature_loss + \
               self.lambda_2 * per_pixel_loss

        return loss.to(device), content_feature_loss, per_pixel_loss

    def tv_regularizer(self, input, beta=2.):
        """
        a total variational regularizer (reduces high frequency structures)
        @param input:
        @param beta:
        @return:
        """
        dy = torch.zeros(input.size())
        dx = torch.zeros(input.size())
        dy[:, 1:, :] = -input[:, :-1, :] + input[:, 1:, :]
        dx[:, :, 1:] = -input[:, :, :-1] + input[:, :, 1:]
        return torch.sum((dx.pow(2) + dy.pow(2)).pow(beta / 2.))


class Decoder(nn.Module):
    """
    the decoder network
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # first block
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu_1_1 = nn.ReLU(inplace=True)

        self.unpool_1 = nn.UpsamplingNearest2d(scale_factor=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.reflecPad_2_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_2_3 = nn.ReLU(inplace=True)

        self.reflecPad_2_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_4 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu_2_4 = nn.ReLU(inplace=True)

        self.unpool_2 = nn.UpsamplingNearest2d(scale_factor=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.unpool_3 = nn.UpsamplingNearest2d(scale_factor=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_4_1 = nn.ReLU(inplace=True)

        self.reflecPad_4_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_2 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, input):
        # first block
        out = self.reflecPad_1_1(input)
        out = self.conv_1_1(out)
        out = self.relu_1_1(out)
        out = self.unpool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)
        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)
        out = self.reflecPad_2_3(out)
        out = self.conv_2_3(out)
        out = self.relu_2_3(out)
        out = self.reflecPad_2_4(out)
        out = self.conv_2_4(out)
        out = self.relu_2_4(out)
        out = self.unpool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)
        out = self.unpool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)
        out = self.reflecPad_4_2(out)
        out = self.conv_4_2(out)

        return out
    
    
        


class Encoder(nn.Module):
    """
    the encoder network
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # first block
        self.conv_1_1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad_1_1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv_1_2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu_1_2 = nn.ReLU(inplace=True)

        self.reflecPad_1_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_1_3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu_1_3 = nn.ReLU(inplace=True)

        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.reflecPad_3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_3 = nn.ReLU(inplace=True)

        self.reflecPad_3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu_3_4 = nn.ReLU(inplace=True)

        self.maxPool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu_4_1 = nn.ReLU(inplace=True)

    def forward(self, input):
        output = {}

        # first block
        out = self.conv_1_1(input)
        out = self.reflecPad_1_1(out)
        out = self.conv_1_2(out)
        out = self.relu_1_2(out)

        output['r11'] = out

        out = self.reflecPad_1_3(out)
        out = self.conv_1_3(out)
        out = self.relu_1_3(out)

        out = self.maxPool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)

        output['r21'] = out

        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)

        out = self.maxPool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)

        output['r31'] = out

        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)

        out = self.reflecPad_3_3(out)
        out = self.conv_3_3(out)
        out = self.relu_3_3(out)

        out = self.reflecPad_3_4(out)
        out = self.conv_3_4(out)
        out = self.relu_3_4(out)

        out = self.maxPool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)

        output['r41'] = out

        return output


def get_pretrained_encoder_decoder_model(configuration, use_list=False, list_index=None):
    """
    - creates an autoencoder model that uses the specified encoder and decoder from the config file
    - if the configuration specifies a list, multiple encoder and decoders can be loaded
    @param configuration: the config file
    @param use_list: boolean determining whether to use a list or not
    @param list_index: the current index of the list
    @return:
    """
    # the encoder
    encoder = Encoder()

    if use_list:
        print('loading encoder from {}'.format(configuration['encoder_model_path_list'][list_index]))
        checkpoint = torch.load(configuration['encoder_model_path_list'][list_index], map_location='cpu')
    else:
        checkpoint = torch.load(configuration['encoder_model_path'], map_location='cpu')
    encoder.load_state_dict(checkpoint)

    # the decoder
    decoder = Decoder()
    if use_list:
        print('loading decoder from {}'.format(configuration['decoder_model_path_list'][list_index]))
        checkpoint = torch.load(configuration['decoder_model_path_list'][list_index], map_location='cpu')
    else:
        checkpoint = torch.load(configuration['decoder_model_path'], map_location='cpu')
    decoder.load_state_dict(checkpoint)

    # loss factor lamda_1
    lambda_1 = configuration['lambda_1']

    # loss factor lamda_2
    lambda_2 = configuration['lambda_2']

    # loss factor lamda_tv
    lambda_tv = configuration['lambda_tv']

    # the model
    encoder_decoder_model = Autoencoder(encoder, decoder, lambda_1, lambda_2, lambda_tv)
    print('the encoder-decoder model: ')
    print(encoder_decoder_model)

    # use the max amount of GPUs possible
    if torch.cuda.device_count() > 1:
        print('Let\'s use', torch.cuda.device_count(), "GPUs!")
        encoder_decoder_model.to(device)
        encoder_decoder_model = nn.DataParallel(encoder_decoder_model)
    else:
        print('Let\'s use the {}'.format(device))

    print('printing whole model params which requires grad')
    for name, param in encoder_decoder_model.named_parameters():
        if param.requires_grad:
            print(name)

    return encoder_decoder_model


def get_encoder_decoder_model(configuration):
    """
    creates an autoencoder model that uses the specified encoder and a randomly initialized decoder
    @param configuration:
    @return:
    """
    encoder_model_path = configuration['encoder_model_path']
    print('using the encoder from {}'.format(encoder_model_path))

    checkpoint = torch.load(encoder_model_path, map_location='cpu')
    print('loaded checkpoint')

    encoder = Encoder()
    print('got encoder model')
    encoder.load_state_dict(checkpoint)
    print('loaded encoder checkpoint')

    for param in encoder.parameters():
        param.requires_grad = False

    decoder = Decoder()
    print('got decoder model')

    # loss factors
    lambda_1 = configuration['lambda_1']
    lambda_2 = configuration['lambda_2']
    lambda_tv = configuration['lambda_tv']

    encoder_decoder_model = Autoencoder(encoder, decoder, lambda_1, lambda_2, lambda_tv)

    # use the max amount of GPUs possible
    if torch.cuda.device_count() > 1:
        print('Let\'s use', torch.cuda.device_count(), "GPUs!")
        encoder_decoder_model.to(device)
        encoder_decoder_model = nn.DataParallel(encoder_decoder_model)
    else:
        print('Let\'s use the {}'.format(device))

    print('printing whole model params which require_grad')
    for name, param in encoder_decoder_model.named_parameters():
        if param.requires_grad:
            print(name)

    return encoder_decoder_model


########################################################################
# loss
########################################################################

class L2Loss(nn.Module):
    """
    simple L_2-Loss
    """
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, input, target):
        return torch.sqrt(torch.mean(torch.abs(input - target).pow(2)))



