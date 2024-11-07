import os
import cv2
import json
import torch
import csv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
import time
from PIL import Image
import glob
import sys 
import scipy.io.wavfile as wav
from scipy import signal
import random
import soundfile as sf

class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        if mode == 'train':
            # json_file = '/mnt/scratch/users/individuals/VGGsound_individual/metadata/train_a_third.json'
            # data_path = '/mnt/scratch/users/individuals/VGGsound_individual/train'
            json_file = '/mnt/scratch/users/sally/VGGsound_individual/metadata/train_practice.json'
            data_path = '/mnt/scratch/users/sally/VGGsound_individual/train_practice'
        else:
            json_file = '/mnt/scratch/users/individuals/VGGsound_individual/metadata/test.json'
            data_path = '/mnt/scratch/users/individuals/VGGsound_individual/test'

        with open(json_file, 'r') as f:
            self.data = json.load(f)['data']

        self.audio_path = os.path.join(data_path, 'sample_audio')
        self.video_path = os.path.join(data_path, 'sample_frames/frame_4')

        
        self.imgSize = args.image_size 
        self.mode = mode
        self.transforms = transforms

        # initialize video transform
        self._init_atransform()  # audio spectrogram의 텐서화 및 정규화를 수행
        self._init_transform() # image 변환 작업을 수행
        #  Retrieve list of audio and video files

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.mode == 'train': # 훈련 모드
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC), # image를 self.imgSize의 1.1배 크기로 확대
                transforms.RandomCrop(self.imgSize), # random crop을 통해 image의 일부분을 임의로 자름
                transforms.RandomHorizontalFlip(), # 50% 확률로 image를 좌우로 뒤집어 데이터 다양성을 더함.
                transforms.CenterCrop(self.imgSize), # 중앙 부분을 잘라냄
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]) # 텐서로 변환 후 정규화
        else: # test 모드
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]) # 텐서로 변환 후 정규화           

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])
        # audio spectrogram data를 tensor 형식으로 변환. -> 표준 편차를 12.0으로 설정하여 정규화
#  

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB') # 주어진 경로에서 image를 읽고 RGB로 변환하여 반환
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.data)  # self.length

    def __getitem__(self, idx):
        # json file에서 audio & video path 가져오기
        
        item = self.data[idx]
        video_id = item['video_id']

        audio_path = os.path.join(self.audio_path, f"{video_id}.wav")
        video_path = os.path.join(self.video_path, f"{video_id}.jpg")

        # image load 및 전처리
        frame = self.img_transform(self._load_frame(video_path))
        frame_ori = np.array(self._load_frame(video_path))
        
        # 오디오 로드 및 전처리
        samples, samplerate = sf.read(audio_path)


        # repeat if audio is too short -> audio 길이가 10초에 미치지 못하면, 필요한 길이만큼 반복하여 samples를 확장
        if samples.shape[0] < samplerate * 10:
            n = int(samplerate * 10 / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate*10] # samples에서 정확히 10초를 잘라내어 resamples에 저장

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples,samplerate, nperseg=512,noverlap=274) # audio spectrogram을 생성
        spectrogram = np.log(spectrogram+ 1e-7) # spectrogram의 값을 log scale로 변환 -> spectrogram에 저장
        spectrogram = self.aid_transform(spectrogram) # spectrogram data를 tensor로 변환하고 정규화
 

        return frame,spectrogram,resamples,video_id,torch.tensor(frame_ori)