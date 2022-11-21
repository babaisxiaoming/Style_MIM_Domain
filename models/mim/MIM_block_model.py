import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalDIM(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.st2 = nn.InstanceNorm2d(3)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.c0 = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.c1 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(9, 3, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)
        self.l0 = nn.Linear(4 * 112 * 112, 128)
        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, Y1, Y2, Y3, M1, M2, M3):
        batch_size = Y1.shape[0]
        M1 = self.up(M1)
        M2 = M2
        M3 = self.pool(M3)
        M = torch.cat((M1, M2, M3), dim=1)
        M = F.relu(self.st2(self.c3(M)))
        M = F.relu(self.st2(self.c4(M)))

        Y1 = self.up(Y1)
        Y3 = self.pool(Y3)
        Y = torch.cat((Y1, Y2, Y3), dim=1)
        Y = F.relu(self.st2(self.c5(Y)))
        h = M.view(batch_size, -1)
        Y = Y.view(batch_size, -1)
        h = torch.cat((Y, h), dim=1)
        h = h.view(batch_size, -1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        score = self.l2(h)

        return score


class LocalDIM(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.st2 = nn.InstanceNorm2d(3)
        self.st1 = nn.InstanceNorm2d(1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.c0 = nn.Conv2d(3 + 32, 64, kernel_size=3, padding=1)
        self.c1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(9, 3, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(12, 3, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.l0 = nn.Linear(1 * 112 * 112, 32)

    def forward(self, Y1, Y2, Y3, M1, M2, M3):
        Y1 = self.up(Y1)
        Y3 = self.pool(Y3)
        Y = torch.cat((Y1, Y2, Y3), dim=1)
        Y = F.relu(self.st2(self.c6(Y)))
        Y = F.relu(self.st1(self.c7(Y)))
        Y = Y.view(Y.shape[0], -1)
        Y = self.l0(Y)
        Y = Y.unsqueeze(-1).unsqueeze(-1)
        Y = Y.expand(-1, -1, 112, 112)

        M1 = self.up(M1)
        M2 = M2
        M3 = self.pool(M3)
        M = torch.cat((M1, M2, M3), dim=1)
        M = F.relu(self.st2(self.c3(M)))

        final = torch.cat((Y, M), dim=1)
        final = F.relu(self.c0(final))
        final = F.relu(self.c1(final))
        final = self.c2(final)
        return final


class PriorDIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.st2 = nn.InstanceNorm2d(12)
        self.c0 = nn.Conv2d(12, 1, kernel_size=3, padding=1)
        self.l0 = nn.Linear(56 * 56, 256)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, Y1, Y2, Y3):
        Y2 = self.pool(Y2)
        Y3 = self.pool2(Y3)
        Y = torch.cat((Y1, Y2, Y3), dim=1)
        Y = F.relu(self.st2(self.c0(Y)))
        Y = Y.view(Y.shape[0], -1)
        h = F.relu(self.l0(Y))
        h = F.relu(self.l1(h))
        output = torch.sigmoid(self.l2(h))

        return output


def MI_JSD(Tp, Tn):
    """
    Input:
        Tp: Discriminator score of positive pair
        Tn: Discriminator score of negative pair
    """
    Ej = -F.softplus(-Tp).mean()
    Em = F.softplus(Tn).mean()
    I = Ej - Em
    return I


class DeepInfoMax(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_MI = GlobalDIM()
        self.local_MI = LocalDIM()
        self.prior_MI = PriorDIM()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, Y1, Y2, Y3, M1, M2, M3):
        # create positive and negative pairs; One image has 1 negative sample here
        M1_prime = torch.cat((M1[1:], M1[0].unsqueeze(0)), dim=0)
        M2_prime = torch.cat((M2[1:], M2[0].unsqueeze(0)), dim=0)
        M3_prime = torch.cat((M3[1:], M3[0].unsqueeze(0)), dim=0)

        if self.alpha != 0:
            Tp = self.global_MI(Y1, Y2, Y3, M1, M2, M3)
            Tn = self.global_MI(Y1, Y2, Y3, M1_prime, M2_prime, M3_prime)
            Ig = MI_JSD(Tp, Tn)
            GLOBAL = self.alpha * Ig
        else:
            GLOBAL = 0

        if self.beta != 0:
            Tp = self.local_MI(Y1, Y2, Y3, M1, M2, M3)
            Tn = self.local_MI(Y1, Y2, Y3, M1_prime, M2_prime, M3_prime)
            Il = MI_JSD(Tp, Tn)
            LOCAL = self.beta * Il
        else:
            LOCAL = 0

        if self.gamma != 0:
            prior1, prior2, prior3 = torch.rand_like(Y1), torch.rand_like(Y2), torch.rand_like(Y3)
            Dp = self.prior_MI(prior1, prior2, prior3)
            Dy = self.prior_MI(Y1, Y2, Y3)
            PRIOR = self.gamma * (torch.log(Dp).mean() + torch.log(1.0 - Dy).mean())
        else:
            PRIOR = 0

        if GLOBAL > 0:
            print("[!] Global value:", GLOBAL)
        if LOCAL > 0:
            print("[!] Local value:", LOCAL)
        if PRIOR > 0:
            print("[!] Prior value:", PRIOR)
        return -(GLOBAL + LOCAL + PRIOR)


if __name__ == '__main__':
    loss_mi = DeepInfoMax(0.5, 1.0, 0.1)

    Y1 = torch.randn((1, 4, 56, 56))
    Y2 = torch.randn((1, 4, 112, 112))
    Y3 = torch.randn((1, 4, 224, 224))

    M1 = torch.randn((1, 3, 56, 56))
    M2 = torch.randn((1, 3, 112, 112))
    M3 = torch.randn((1, 3, 224, 224))

    print(loss_mi(Y1, Y2, Y3, M1, M2, M3).mean())
