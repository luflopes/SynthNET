import os
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from typing import Tuple
from src.utils import (
    EarlyStopping,
    cross_diff_filter,
    fft_peak_feats,
    fft2D
)
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score
)
from pathlib import Path



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.conv(spatial_attention)
        return x * self.sigmoid(spatial_attention)


class SynthNET(nn.Module):
    def __init__(self, ckp:Path=None, input_size:tuple=(3, 224, 224), device:str=None, freeze_conv:bool=True):
        super(SynthNET, self).__init__()

        self.num_fourier_peaks = 135
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.mobilenet_v3_small(
            input_size=input_size,
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        
        self.inverted_res_block_1 = nn.Sequential(*list(model.features.children())[:4])
        self.fourier_proj_1 = nn.Linear(self.num_fourier_peaks, 24)
        self.attention_1 = SpatialAttention(kernel_size=7)

        self.inverted_res_block_2 = nn.Sequential(*list(model.features.children())[4:8])
        self.fourier_proj_2 = nn.Linear(self.num_fourier_peaks, 48)
        self.attention_2 = SpatialAttention(kernel_size=7)

        self.inverted_res_block_3 = nn.Sequential(*list(model.features.children())[8:-1])
        self.fourier_proj_3 = nn.Linear(self.num_fourier_peaks, 96)
        self.attention_3 = SpatialAttention(kernel_size=7)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=4704, out_features=576, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=576, out_features=1, bias=True)
        )

        if ckp is not None:
            if os.path.exists(ckp):
                print("Checkpoint loaded")
                self.load_state_dict(torch.load(ckp, weights_only=True))
            else:
                raise FileNotFoundError("Checkpoint file was not found in direcory")
        else:
            self.initialize_weights()

        if freeze_conv:
            for layer in [self.inverted_res_block_1, self.inverted_res_block_2, self.inverted_res_block_3]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.to(self.device)


    def initialize_weights(self):
        # Initialize only the weights of the custom layers (not the pretrained MobileNet layers)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and ('fourier_proj' in name or 'classifier' in name):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) and 'attention' in name:
                nn.init.normal_(m.weight, mean=0, std=0.01)


    def adjust_learning_rate(self, optimizer, min_lr=1e-6):
        for param_group in optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True


    def forward(self, img, peaks):

        x = self.inverted_res_block_1(img)
        peaks_proj_1 = torch.relu(self.fourier_proj_1(peaks))
        x = x + peaks_proj_1.unsqueeze(-1).unsqueeze(-1)
        x = self.attention_1(x)

        x = self.inverted_res_block_2(x)
        peaks_proj_2 = torch.relu(self.fourier_proj_2(peaks))
        x = x + peaks_proj_2.unsqueeze(-1).unsqueeze(-1)
        x = self.attention_2(x)

        x = self.inverted_res_block_3(x)
        peaks_proj_3 = torch.relu(self.fourier_proj_3(peaks))
        x = x + peaks_proj_3.unsqueeze(-1).unsqueeze(-1)
        x = self.attention_3(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x


    def make_dirs(self, root_dir):
        self.weights_path = os.path.join(root_dir, "weights")
        self.imgs_path = os.path.join(root_dir, "images")
        self.metrics_path = os.path.join(root_dir, "metrics")

        for dir in [self.weights_path, self.imgs_path, self.metrics_path]:
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)


    def train_model(self, root_dir, dataloader, num_epochs, optimizer, criterion, threshold=0.5):
        root_dir = root_dir or os.getcwd()
        self.make_dirs(root_dir)
        self.writer = SummaryWriter(self.metrics_path)

        self.train()

        train_size = len(dataloader["train"].dataset)
        eval_size = len(dataloader["eval"].dataset)
        criterion = criterion.to(self.device)
        early_stopping = None

        print()

        for epoch in range(num_epochs):
            train_running_loss = 0.0
            eval_running_loss = 0.0

            # Acumulate labels and predictions
            train_preds_all = []
            train_labels_all = []
            eval_preds_all = []
            eval_labels_all = []

            # Training
            self.train()
            for train_images, train_labels in dataloader["train"]:
                train_images = train_images.to(self.device)
                train_labels = train_labels.unsqueeze(1).to(torch.float32).to(self.device)

                optimizer.zero_grad()

                # Forward: Apply filter, extract peaks and get predictions
                train_images = cross_diff_filter(train_images)
                fft = fft2D(train_images)
                peaks = fft_peak_feats(fft).to(self.device)
                train_preds = self.forward(train_images, peaks)
                
                # Calculate loss and do backpropagations
                loss = criterion(train_preds, train_labels)
                loss.backward()
                optimizer.step()

                train_running_loss += loss.item() * train_images.size(0)

                train_preds_all.extend(train_preds.detach().cpu().numpy())
                train_labels_all.extend(train_labels.detach().cpu().numpy())

            # Calculate training metrics
            train_loss = train_running_loss / train_size
            train_accuracy = balanced_accuracy_score(train_labels_all, np.array(train_preds_all) > threshold)
            train_auc = roc_auc_score(train_labels_all, train_preds_all)

            # Validation
            self.eval()
            for eval_images, eval_labels in dataloader["eval"]:
                eval_images = eval_images.to(self.device)
                eval_labels = eval_labels.unsqueeze(1).to(torch.float32).to(self.device)

                with torch.no_grad():
                    eval_images = cross_diff_filter(eval_images)
                    fft = fft2D(eval_images)
                    peaks = fft_peak_feats(fft).to(self.device)
                    eval_preds = self.forward(eval_images, peaks)
                    loss = criterion(eval_preds, eval_labels)

                eval_running_loss += loss.item() * eval_images.size(0)

                eval_preds_all.extend(eval_preds.detach().cpu().numpy())
                eval_labels_all.extend(eval_labels.detach().cpu().numpy())

            # Calculate validation metrics
            eval_loss = eval_running_loss / eval_size
            eval_accuracy = balanced_accuracy_score(eval_labels_all, np.array(eval_preds_all) > threshold)
            eval_auc = roc_auc_score(eval_labels_all, eval_preds_all)
            
            # Register metrics
            self.writer.add_scalar("Train_Loss", train_loss, epoch)
            self.writer.add_scalar("Train_Accuracy", train_accuracy, epoch)
            self.writer.add_scalar("Train_AUC", train_auc, epoch)
            self.writer.add_scalar("Val_Loss", eval_loss, epoch)
            self.writer.add_scalar("Val_Accuracy", eval_accuracy, epoch)
            self.writer.add_scalar("Val_AUC", eval_auc, epoch)

            # Save model for each epoch
            torch.save(self.state_dict(), os.path.join(self.weights_path, f"synthnet-epoch-{epoch + 1}.pth"))

            print(
                f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val. Loss: {eval_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.4f} | Val. Accuracy: {eval_accuracy:.4f} | "
                f"Train AUC: {train_auc:.4f} | Val AUC: {eval_auc:.4f}"
            )

            # Early Stopping
            if early_stopping is None:
                early_stopping = EarlyStopping(init_score=eval_loss, patience=5, delta=0.001, verbose=False)
            else:
                if early_stopping(eval_loss):
                    best_model_path = os.path.join(self.weights_path, f"synthnet-best.pth")
                    print(f"\nEpoch [{epoch+1}/{num_epochs}] Saved best model at: {best_model_path}\n", flush=True)
                    torch.save(self.state_dict(), best_model_path)
                if early_stopping.early_stop:
                    cont_train = self.adjust_learning_rate(optimizer)
                    if cont_train:
                        print(f"\nEpoch [{epoch+1}/{num_epochs}] Learning rate dropped by 10, continue training ...\n", flush=True)
                        early_stopping.reset_counter()
                    else:
                        print(f"\nEpoch [{epoch+1}/{num_epochs}] Early stopping.\n", flush=True)
                        break



    def predict(self, images):
        """
        Performs a prediction on images.
        Args:
        images (torch.Tensor): Image in format [batch_size, 3, 224, 224] already processed.
        Returns:
        torch.Tensor: Model output.
        """
        with torch.no_grad():
            images = images.to(self.device)
            images = cross_diff_filter(images)
            fft = fft2D(images)
            peaks = fft_peak_feats(fft).to(self.device)
            output = self.forward(images, peaks)
        return output