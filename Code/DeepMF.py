# Copyright (C) 2024 ETH Zurich. All rights reserved.   
# Author: Carlos Santos, ETH Zurich           

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.   
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.   
# SPDX-License-Identifier: Apache-2.0


# Imports
import torch
from torch import nn
import torch.nn.functional as F

#############################################################
# Network definitions
#############################################################

## Base implementation
class Encoder(nn.Module):
    """" Encoder of the Deep-MF model """

    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 6, kernel_size=201, stride=1, padding=25)
        self.conv2 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
        self.conv3 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
        self.conv4 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.conv4(x))
        x = self.dropout(x)
        return x
    
class Decoder(nn.Module):
    """ Decoder of the Deep-MF model """

    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
        self.deconv3 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
        self.deconv4 = nn.ConvTranspose1d(in_channels = 6, out_channels = 1, kernel_size=201, stride=1, padding=25)
    
    def forward(self, x):
        x = torch.sigmoid(self.deconv1(x))
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x
    
class RPeakClassifier(nn.Module):
    """ R Peak Classifier of the Deep-MF model"""

    def __init__(self):
        super(RPeakClassifier, self).__init__()
        self.classifier_conv = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size = 51, stride = 1)
        self.fc = nn.Linear(in_features = 250*6*1, out_features = 500)

    def forward(self, x):
        x = torch.sigmoid(self.classifier_conv(x))
        x = torch.flatten(x, start_dim = 1)
        x = self.fc(x)
        return x
    
class DeepMFEncoderDecoder(nn.Module):
    """ Encoder-Decoder architecture """

    def __init__(self, in_channels):
        super(DeepMFEncoderDecoder, self).__init__()
        self.encoder = Encoder(in_channels) # Encoder
        self.decoder = Decoder() # Decoder
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def write_weights(self, fname):
        """ Store learned weights of encoder-decoder """
        torch.save(self.state_dict(), fname)

class DeepMFClassifier(nn.Module):
    """ Deep-MF Classifier model """

    def __init__(self, in_channels):
        super(DeepMFClassifier, self).__init__()
        self.encoder = Encoder(in_channels) # Encoder
        self.classifier = RPeakClassifier() # Classifier
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
    def write_weights(self, fname):
        """ Store learned weights of encoder-classifier """
        torch.save(self.state_dict(), fname)


# ## V2: BN and tanh activation
# class Encoder_v2(nn.Module):
#     """" Encoder of the Deep-MF model """

#     def __init__(self, in_channels):
#         super(Encoder_v2, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 6, kernel_size=201, stride=1, padding=25)
#         self.conv2 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
#         self.conv3 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
#         self.conv4 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1)

#         self.bn1 = nn.BatchNorm1d(6)
#         self.bn2 = nn.BatchNorm1d(6)
#         self.bn3 = nn.BatchNorm1d(6)
#         self.bn4 = nn.BatchNorm1d(6)

#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn4(self.conv4(x))) # substitute sigmoid by relu
#         x = self.dropout(x)
#         return x

# class Decoder_v2(nn.Module):
#     """ Decoder of the Deep-MF model """

#     def __init__(self):
#         super(Decoder_v2, self).__init__()
#         self.deconv1 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1)
#         self.deconv2 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
#         self.deconv3 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
#         self.deconv4 = nn.ConvTranspose1d(in_channels = 6, out_channels = 1, kernel_size=201, stride=1, padding=25)
        
#         self.bn1 = nn.BatchNorm1d(6) # include BN
#         self.bn2 = nn.BatchNorm1d(6)
#         self.bn3 = nn.BatchNorm1d(6)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.deconv1(x)))
#         x = F.relu(self.bn2(self.deconv2(x)))
#         x = F.relu(self.bn3(self.deconv3(x)))
#         x = F.tanh(self.deconv4(x)) # tanh activation
#         return x
    
# class RPeakClassifier_v2(nn.Module):
#     """ R Peak Classifier of the Deep-MF model"""

#     def __init__(self):
#         super(RPeakClassifier_v2, self).__init__()
#         self.classifier_conv = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size = 51, stride = 1)
#         self.fc = nn.Linear(in_features = 250*6*1, out_features = 500)

#     def forward(self, x):
#         x = F.relu(self.classifier_conv(x)) # substitute sigmoid by relu
#         x = torch.flatten(x, start_dim = 1)
#         x = torch.sigmoid(self.fc(x)) # sigmoid activation
#         return x
    
# class DeepMFEncoderDecoder_v2(nn.Module):
#     """ Encoder-Decoder architecture """

#     def __init__(self, in_channels):
#         super(DeepMFEncoderDecoder_v2, self).__init__()
#         self.encoder = Encoder_v2(in_channels) # Encoder
#         self.decoder = Decoder_v2() # Decoder
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
#     def write_weights(self, fname):
#         """ Store learned weights of encoder-decoder """
#         torch.save(self.state_dict(), fname)

# class DeepMFClassifier_v2(nn.Module):
#     """ Deep-MF Classifier model """

#     def __init__(self, in_channels):
#         super(DeepMFClassifier_v2, self).__init__()
#         self.encoder = Encoder_v2(in_channels) # Encoder
#         self.classifier = RPeakClassifier_v2() # Classifier
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.classifier(x)
#         return x
    
#     def write_weights(self, fname):
#         """ Store learned weights of encoder-classifier """
#         torch.save(self.state_dict(), fname)

# ## V3: larger intermediate kernel
# class Encoder_v3(nn.Module):
#     """" Encoder of the Deep-MF model """

#     def __init__(self, in_channels):
#         super(Encoder_v3, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 6, kernel_size=201, stride=1, padding=25)
#         self.conv2 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=121, stride=1, padding=60)
#         self.conv3 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=121, stride=1, padding=60)
#         self.conv4 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=121, stride=1, padding=35)

#         self.bn1 = nn.BatchNorm1d(6)
#         self.bn2 = nn.BatchNorm1d(6)
#         self.bn3 = nn.BatchNorm1d(6)
#         self.bn4 = nn.BatchNorm1d(6)

#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = self.dropout(x)
#         return x

# class Decoder_v3(nn.Module):
#     """ Decoder of the Deep-MF model """

#     def __init__(self):
#         super(Decoder_v3, self).__init__()
#         self.deconv1 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=121, stride=1, padding=35)
#         self.deconv2 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=121, stride=1, padding=60)
#         self.deconv3 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=121, stride=1, padding=60)
#         self.deconv4 = nn.ConvTranspose1d(in_channels = 6, out_channels = 1, kernel_size=201, stride=1, padding=25)

#         self.bn1 = nn.BatchNorm1d(6)
#         self.bn2 = nn.BatchNorm1d(6)
#         self.bn3 = nn.BatchNorm1d(6)
    
#     def forward(self, x):
#         x = F.relu(self.bn1(self.deconv1(x)))
#         x = F.relu(self.bn2(self.deconv2(x)))
#         x = F.relu(self.bn3(self.deconv3(x)))
#         x = F.tanh(self.deconv4(x))
#         return x
    
# class RPeakClassifier_v3(nn.Module):
#     """ R Peak Classifier of the Deep-MF model"""

#     def __init__(self):
#         super(RPeakClassifier_v3, self).__init__()
#         self.classifier_conv = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size = 51, stride = 1)
#         self.fc = nn.Linear(in_features = 250*6*1, out_features = 500)

#     def forward(self, x):
#         x = F.relu(self.classifier_conv(x)) # substitute sigmoid by relu
#         x = torch.flatten(x, start_dim = 1)
#         x = torch.sigmoid(self.fc(x)) # sigmoid activation
#         return x
    
# class DeepMFEncoderDecoder_v3(nn.Module):
#     """ Encoder-Decoder architecture """

#     def __init__(self, in_channels):
#         super(DeepMFEncoderDecoder_v3, self).__init__()
#         self.encoder = Encoder_v3(in_channels) # Encoder
#         self.decoder = Decoder_v3() # Decoder
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
#     def write_weights(self, fname):
#         """ Store learned weights of encoder-decoder """
#         torch.save(self.state_dict(), fname)

# class DeepMFClassifier_v3(nn.Module):
#     """ Deep-MF Classifier model """

#     def __init__(self, in_channels):
#         super(DeepMFClassifier_v3, self).__init__()
#         self.encoder = Encoder_v3(in_channels) # Encoder
#         self.classifier = RPeakClassifier_v3() # Classifier
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.classifier(x)
#         return x
    
#     def write_weights(self, fname):
#         """ Store learned weights of encoder-classifier """
#         torch.save(self.state_dict(), fname)

# ## V4: BN and tanh activation changing channels-
# class Encoder_v4(nn.Module):
#     """" Encoder of the Deep-MF model """

#     def __init__(self, in_channels):
#         super(Encoder_v4, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 4, kernel_size=201, stride=1, padding=25)
#         self.conv2 = nn.Conv1d(in_channels = 4, out_channels = 8, kernel_size=51, stride=1, padding=25)
#         self.conv3 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size=51, stride=1)

#         self.bn1 = nn.BatchNorm1d(4)
#         self.bn2 = nn.BatchNorm1d(8)
#         self.bn3 = nn.BatchNorm1d(16)

#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.dropout(x)
#         return x

# class Decoder_v4(nn.Module):
#     """ Decoder of the Deep-MF model """

#     def __init__(self):
#         super(Decoder_v4, self).__init__()
#         self.deconv1 = nn.ConvTranspose1d(in_channels = 16, out_channels = 8, kernel_size=51, stride=1)
#         self.deconv2 = nn.ConvTranspose1d(in_channels = 8, out_channels = 4, kernel_size=51, stride=1, padding=25)
#         self.deconv3 = nn.ConvTranspose1d(in_channels = 4, out_channels = 1, kernel_size=201, stride=1, padding=25)
        
#         self.bn1 = nn.BatchNorm1d(8) # include BN
#         self.bn2 = nn.BatchNorm1d(4)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.deconv1(x)))
#         x = F.relu(self.bn2(self.deconv2(x)))
#         x = F.tanh(self.deconv3(x)) # tanh activation
#         return x
    
# class RPeakClassifier_v4(nn.Module):
#     """ R Peak Classifier of the Deep-MF model"""

#     def __init__(self):
#         super(RPeakClassifier_v4, self).__init__()
#         self.classifier_conv = nn.Conv1d(in_channels = 16, out_channels = 6, kernel_size = 51, stride = 1)
#         self.fc = nn.Linear(in_features = 250*6*1, out_features = 500)

#     def forward(self, x):
#         x = F.relu(self.classifier_conv(x)) # substitute sigmoid by relu
#         x = torch.flatten(x, start_dim = 1)
#         x = torch.sigmoid(self.fc(x)) # sigmoid activation
#         return x
    
# class DeepMFEncoderDecoder_v4(nn.Module):
#     """ Encoder-Decoder architecture """

#     def __init__(self, in_channels):
#         super(DeepMFEncoderDecoder_v4, self).__init__()
#         self.encoder = Encoder_v4(in_channels) # Encoder
#         self.decoder = Decoder_v4() # Decoder
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
#     def write_weights(self, fname):
#         """ Store learned weights of encoder-decoder """
#         torch.save(self.state_dict(), fname)

# class DeepMFClassifier_v4(nn.Module):
#     """ Deep-MF Classifier model """

#     def __init__(self, in_channels):
#         super(DeepMFClassifier_v4, self).__init__()
#         self.encoder = Encoder_v4(in_channels) # Encoder
#         self.classifier = RPeakClassifier_v4() # Classifier
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.classifier(x)
#         return x
    
#     def write_weights(self, fname):
#         """ Store learned weights of encoder-classifier """
#         torch.save(self.state_dict(), fname)

# ## V5: V2 + convolutional classifier
# class Encoder_v5(nn.Module):
#     """" Encoder of the Deep-MF model """

#     def __init__(self, in_channels):
#         super(Encoder_v5, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 6, kernel_size=201, stride=1, padding=25)
#         self.conv2 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
#         self.conv3 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
#         self.conv4 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1)

#         self.bn1 = nn.BatchNorm1d(6)
#         self.bn2 = nn.BatchNorm1d(6)
#         self.bn3 = nn.BatchNorm1d(6)
#         self.bn4 = nn.BatchNorm1d(6)

#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.dropout(x)
#         x = F.relu(self.bn4(self.conv4(x))) # substitute sigmoid by relu
#         x = self.dropout(x)
#         return x

# class Decoder_v5(nn.Module):
#     """ Decoder of the Deep-MF model """

#     def __init__(self):
#         super(Decoder_v5, self).__init__()
#         self.deconv1 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1)
#         self.deconv2 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
#         self.deconv3 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
#         self.deconv4 = nn.ConvTranspose1d(in_channels = 6, out_channels = 1, kernel_size=201, stride=1, padding=25)
        
#         self.bn1 = nn.BatchNorm1d(6) # include BN
#         self.bn2 = nn.BatchNorm1d(6)
#         self.bn3 = nn.BatchNorm1d(6)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.deconv1(x)))
#         x = F.relu(self.bn2(self.deconv2(x)))
#         x = F.relu(self.bn3(self.deconv3(x)))
#         x = F.tanh(self.deconv4(x)) # tanh activation
#         return x
    
# class RPeakClassifier_v5(nn.Module): # same as Decoder_v5 with sigmoid and squeeze
#     """ R Peak Classifier of the Deep-MF model"""

#     def __init__(self):
#         super(RPeakClassifier_v5, self).__init__()
#         self.deconv1 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1)
#         self.deconv2 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
#         self.deconv3 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
#         self.deconv4 = nn.ConvTranspose1d(in_channels = 6, out_channels = 1, kernel_size=201, stride=1, padding=25)
        
#         self.bn1 = nn.BatchNorm1d(6) # include BN
#         self.bn2 = nn.BatchNorm1d(6)
#         self.bn3 = nn.BatchNorm1d(6)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.deconv1(x)))
#         x = F.relu(self.bn2(self.deconv2(x)))
#         x = F.relu(self.bn3(self.deconv3(x)))
#         x = F.sigmoid(self.deconv4(x)) # sigmoid activation
#         x = x.squeeze() # Handle ecg dimension
#         return x
    
# class DeepMFEncoderDecoder_v5(nn.Module): # same as DeepMFEncoderDecoder_v2
#     """ Encoder-Decoder architecture """

#     def __init__(self, in_channels):
#         super(DeepMFEncoderDecoder_v5, self).__init__()
#         self.encoder = Encoder_v5(in_channels) # Encoder
#         self.decoder = Decoder_v5() # Decoder
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
#     def write_weights(self, fname):
#         """ Store learned weights of encoder-decoder """
#         torch.save(self.state_dict(), fname)

# class DeepMFClassifier_v5(nn.Module):
#     """ Deep-MF Classifier model """

#     def __init__(self, in_channels):
#         super(DeepMFClassifier_v5, self).__init__()
#         self.encoder = Encoder_v5(in_channels) # Encoder
#         self.classifier = RPeakClassifier_v5() # Convolutional classifier
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.classifier(x)
#         return x
    
#     def write_weights(self, fname):
#         """ Store learned weights of encoder-classifier """
#         torch.save(self.state_dict(), fname)


## V6: BN and tanh activation - 2-convlayer classifier
class Encoder_v6(nn.Module):
    """" Encoder of the Deep-MF model """

    def __init__(self, in_channels):
        super(Encoder_v6, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 6, kernel_size=201, stride=1, padding=25)
        self.conv2 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
        self.conv3 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
        self.conv4 = nn.Conv1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1)

        self.bn1 = nn.BatchNorm1d(6)
        self.bn2 = nn.BatchNorm1d(6)
        self.bn3 = nn.BatchNorm1d(6)
        self.bn4 = nn.BatchNorm1d(6)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x))) # substitute sigmoid by relu
        x = self.dropout(x)
        return x

class Decoder_v6(nn.Module):
    """ Decoder of the Deep-MF model """

    def __init__(self):
        super(Decoder_v6, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
        self.deconv3 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1, padding=25)
        self.deconv4 = nn.ConvTranspose1d(in_channels = 6, out_channels = 1, kernel_size=201, stride=1, padding=25)
        
        self.bn1 = nn.BatchNorm1d(6) # include BN
        self.bn2 = nn.BatchNorm1d(6)
        self.bn3 = nn.BatchNorm1d(6)

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.tanh(self.deconv4(x)) # tanh activation
        return x
    
class RPeakClassifier_v6(nn.Module):
    """ R Peak Classifier of the Deep-MF model"""

    def __init__(self):
        super(RPeakClassifier_v6, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channels = 6, out_channels = 6, kernel_size=51, stride=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = 6, out_channels = 1, kernel_size=201, stride=1, padding=25)
        
        self.bn1 = nn.BatchNorm1d(6) # include BN

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = torch.sigmoid(self.deconv2(x)) # tanh activation
        x = x.squeeze(1) # Handle ecg dimension
        return x
    
class DeepMFMiniEncoderDecoder(nn.Module):
    """ Encoder-Decoder architecture """

    def __init__(self, in_channels):
        super(DeepMFMiniEncoderDecoder, self).__init__()
        self.encoder = Encoder_v6(in_channels) # Encoder
        self.decoder = Decoder_v6() # Decoder
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def write_weights(self, fname):
        """ Store learned weights of encoder-decoder """
        torch.save(self.state_dict(), fname)

class DeepMFMiniClassifier(nn.Module):
    """ Deep-MF Classifier model """

    def __init__(self, in_channels):
        super(DeepMFMiniClassifier, self).__init__()
        self.encoder = Encoder_v6(in_channels) # Encoder
        self.classifier = RPeakClassifier_v6() # Classifier
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
    def write_weights(self, fname):
        """ Store learned weights of encoder-classifier """
        torch.save(self.state_dict(), fname)