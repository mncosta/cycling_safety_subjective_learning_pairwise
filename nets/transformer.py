import sys
import torch.nn as nn
import torch


class Transformer(nn.Module):

    def __init__(self, backbone, model, num_classes=2):
        super(Transformer, self).__init__()
        self.model = model

        # Set backbone model
        self.transformer = backbone
        print('\n', '='*20, '\n')
        # Ranking Subnetwork
        self.rank_fc_1 = nn.Linear(self.transformer.head.out_features, 4096)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.rank_fc_out = nn.Linear(4096, 1)

        # Classification Subnetwork
        self.cross_fc_1 = nn.Linear(self.transformer.head.out_features * 2, 512)
        self.relu_1 = nn.ReLU()
        self.drop_1 = nn.Dropout(0.3)
        self.cross_fc_2 = nn.Linear(512, 512)
        self.relu_2 = nn.ReLU()
        self.drop_2 = nn.Dropout(0.3)
        self.cross_fc_3 = nn.Linear(512, num_classes)

    # 'rsscnn',  # Classification + Ranking Loss
    # 'sscnn',  # Classification Loss Only
    # 'rcnn',  # Ranking Loss Only

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
        x = self.transformer(batch)
        x = self.rank_fc_1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.rank_fc_out(x)

        return {
            'output': x
        }

    def single_forward_fusion(self, batch_left, batch_right):
        # Left branch
        x_left = self.transformer(batch_left)

        # Right branch
        x_right = self.transformer(batch_right)

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
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

    net = Transformer(model, 'rsscnn', num_classes=3)
    x = torch.randn([3, 224, 224]).unsqueeze(0)
    y = torch.randn([3, 224, 224]).unsqueeze(0)
    fwd = net(x, y)
    print(fwd)