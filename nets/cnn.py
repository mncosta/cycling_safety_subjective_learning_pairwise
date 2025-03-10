import sys

import torchvision.models as models
import torch.nn as nn
import torch


class CNN(nn.Module):

    def __init__(self, backbone, model, finetune=False, num_classes=2):
        super(CNN, self).__init__()
        try:
            self.cnn = backbone(weights='DEFAULT').features
        except AttributeError:
            self.cnn = nn.Sequential(*list(backbone(weights='DEFAULT').children())[:-1])
        if not finetune:
            for param in self.cnn.parameters():  # freeze cnn params
                param.requires_grad = False

        # Save model typology
        self.model = model

        x = torch.randn([3, 244, 244]).unsqueeze(0)
        output_size = self.cnn(x).size()
        self.dims = output_size[1] * 2
        self.cnn_size = output_size

        # Ranking Subnetwork
        self.rank_fc_1 = nn.Linear(self.cnn_size[1] * self.cnn_size[2] * self.cnn_size[3], 4096)
        self.rank_fc_out = nn.Linear(4096, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

        # Classification Subnetwork
        self.cross_fc_1 = nn.Linear((self.cnn_size[1] * self.cnn_size[2] * self.cnn_size[3]) * 2, 512)
        self.relu_1 = nn.ReLU()
        self.drop_1 = nn.Dropout(0.3)
        self.cross_fc_2 = nn.Linear(512, 512)
        self.relu_2 = nn.ReLU()
        self.drop_2 = nn.Dropout(0.3)
        self.cross_fc_3 = nn.Linear(512, num_classes)

    def forward(self, left_batch, right_batch):
        # Ranking Model Only
        if self.model == 'rcnn':
            return {
                'left': self.single_forward_ranking(left_batch),
                'right': self.single_forward_ranking(right_batch),
            }

        # Classification Model Only
        elif self.model == 'sscnn':
            return {
                'logits': self.single_forward_fusion(left_batch, right_batch)
            }

        # Classification + Ranking Model
        elif self.model == 'rsscnn':
            return {
                'left': self.single_forward_ranking(left_batch),
                'right': self.single_forward_ranking(right_batch),
                'logits': self.single_forward_fusion(left_batch, right_batch),
            }

        else:
            print('Invalid model specification. Aborting.')
            sys.exit(-1)

    def single_forward_ranking(self, batch):
        batch_size = batch.size()[0]
        x = self.cnn(batch)

        x = x.reshape(batch_size, self.cnn_size[1] * self.cnn_size[2] * self.cnn_size[3])
        x = self.rank_fc_1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.rank_fc_out(x)

        return {
            'output': x
        }

    def single_forward_fusion(self, batch_left, batch_right):
        # Left branch
        batch_size = batch_left.size()[0]
        x_left = self.cnn(batch_left)
        x_left = x_left.reshape(batch_size, self.cnn_size[1] * self.cnn_size[2] * self.cnn_size[3])

        # Right branch
        batch_size = batch_right.size()[0]
        x_right = self.cnn(batch_right)
        x_right = x_right.reshape(batch_size, self.cnn_size[1] * self.cnn_size[2] * self.cnn_size[3])

        # ================================================================ #
        # Fusion Network
        # ================================================================ #
        # Concatenate left and right outputs of siamese network
        x = torch.cat((x_left, x_right), 1)
        x = self.cross_fc_1(x)
        x = self.relu_1(x)
        x = self.drop_1(x)
        x = self.cross_fc_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)
        x = self.cross_fc_3(x)

        return {
            'output': x
        }


if __name__ == '__main__':
    net = CNN(backbone=models.resnet50, model='rcnn')
    x = torch.randn([3, 244, 244]).unsqueeze(0)
    y = torch.randn([3, 244, 244]).unsqueeze(0)
    fwd = net(x, y)
    print(fwd)
