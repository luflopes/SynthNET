import torch
from torch import nn


def fft_peak_feats(fft2d: torch.Tensor) -> torch.Tensor:
    """Locate peaks in a squared image"""

    height, width = fft2d.shape[-2], fft2d.shape[-1]
    if height != width:
        raise ValueError("The MxN dimensions of the image must be equal")
    
    locs_x = torch.arange(width // 2, width, width // 8)
    locs_y = torch.arange(0, height, height // 8)
    locs_x = torch.cat([locs_x, torch.tensor([width - 1])])
    locs_y = torch.cat([locs_y, torch.tensor([height - 1])])
    locs_x, locs_y = torch.meshgrid(locs_x, locs_y, indexing='xy')
    locs = torch.stack([locs_x, locs_y], dim=-2).reshape(-1, 2)

    peaks = fft2d[..., locs[:, 1], locs[:, 0]]
    return peaks.view(peaks.size(0), -1)


def fft2D(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected input with 4 dimensions, got {x.dim()} dimensions.")
    
    _, _, height, width = x.shape
    fft_result = torch.fft.fft2(x, dim=(-2, -1))
    #fft_shifted = torch.fft.fftshift(fft_result)
    fft_magnitude = torch.abs(fft_result)
    num_pixels = height * width
    fft_magnitude_normalized = fft_magnitude / num_pixels
    return fft_magnitude_normalized


def cross_diff_filter(x: torch.Tensor) -> torch.Tensor:
    x_padded = nn.functional.pad(x, (0, 1, 0, 1), mode='replicate')  # Padding à direita e embaixo
    diagonal1 = x_padded[:, :, :-1, :-1] + x_padded[:, :, 1:, 1:]
    diagonal2 = x_padded[:, :, :-1, 1:] + x_padded[:, :, 1:, :-1]
    cross_difference = torch.abs(diagonal1 - diagonal2)
    return cross_difference



class EarlyStopping:
    '''                                        
    Copyright 2024 Image Processing Research Group of University Federico
    II of Naples ('GRIP-UNINA'). All rights reserved.
                            
    Licensed under the Apache License, Version 2.0 (the "License");       
    you may not use this file except in compliance with the License. 
    You may obtain a copy of the License at                    
                                            
    http://www.apache.org/licenses/LICENSE-2.0
                                                        
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,    
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
    See the License for the specific language governing permissions and
    limitations under the License.
    ''' 
    def __init__(self, init_score=None, patience=1, verbose=False, delta=0):
        self.best_score = init_score
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.count_down = self.patience
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            if self.verbose:
                print(f'Score set to {score:.6f}.')
            self.best_score = score
            self.count_down = self.patience
            return True
        elif score <= self.best_score + self.delta:
            self.count_down -= 1
            if self.verbose:
                print(f'EarlyStopping count_down: {self.count_down} on {self.patience}')
            if self.count_down <= 0:
                self.early_stop = True
            return False
        else:
            if self.verbose:
                print(f'Score increased from ({self.best_score:.6f} to {score:.6f}).')
            self.best_score = score
            self.count_down = self.patience
            return True

    def reset_counter(self):
        self.count_down = self.patience
        self.early_stop = False
