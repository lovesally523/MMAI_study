import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import soundfile as sf
import random
import matplotlib.pyplot as plt
from IPython.display import Audio
from torchvision import transforms

# PlacesAudio Dataset
class PlacesAudioDataset(Dataset):
    def __init__(self, json_file, image_base_path, audio_base_path, transform=None):
        self.transform = transform
        self.image_base_path = image_base_path
        self.audio_base_path = audio_base_path
        
        # JSON 파일 읽기
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.data = data['data']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 이미지 경로 설정
        image_path = os.path.join(self.image_base_path, item['image'])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        
        # 오디오 경로 설정
        audio_path = os.path.join(self.audio_base_path, item['wav'])
        audio, sr = sf.read(audio_path)
        
        return image, audio, sr

# VGGSound Dataset
class VGGSoundDataset(Dataset):
    def __init__(self, json_file, base_path, transform=None):
        self.transform = transform
        self.base_path = base_path
        
        # JSON 파일 읽기
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.data = data['data']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 랜덤 프레임 선택 (0~9 중 랜덤 선택)
        frame_id = random.randint(0, 9)
        frame_path = os.path.join(self.base_path, item['video_path'], f'frame_{frame_id}.jpg')
        image = Image.open(frame_path)
        if self.transform:
            image = self.transform(image)
        
        # 오디오 경로 설정
        audio_path = os.path.join(self.base_path, item['wav'])
        audio, sr = sf.read(audio_path)
        
        return image, audio, sr

# 이미지 및 오디오 시각화 함수
def visualize(image, audio, sr):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    display(Audio(audio, rate=sr))

# Dataloader 설정 (PlacesAudio)
placesaudio_dataset = PlacesAudioDataset(
    json_file='/path/to/val.json',
    image_base_path='/path/to/images',
    audio_base_path='/path/to/audios',
    transform=transforms.ToTensor()
)

placesaudio_loader = DataLoader(placesaudio_dataset, batch_size=1, shuffle=True)

# Dataloader 설정 (VGGSound)
vggsound_dataset = VGGSoundDataset(
    json_file='/path/to/test.json',
    base_path='/path/to/dataset',
    transform=transforms.ToTensor()
)

vggsound_loader = DataLoader(vggsound_dataset, batch_size=1, shuffle=True)

# PlacesAudio 데이터셋 시각화
for image, audio, sr in placesaudio_loader:
    visualize(image.squeeze(0).permute(1, 2, 0), audio, sr)
    break

# VGGSound 데이터셋 시각화
for image, audio, sr in vggsound_loader:
    visualize(image.squeeze(0).permute(1, 2, 0), audio, sr)
    break
