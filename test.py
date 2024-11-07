import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
import json
import argparse
import csv
from model import AVENet
from DatasetLoader import GetAudioVideoDataset
import cv2
from sklearn.metrics import auc
from PIL import Image


def get_arguments():
    parser = argparse.ArgumentParser()
    # 1. 사용할 testset의 이름 지정 (flickr or vggss)
    parser.add_argument('--testset',default='vggss',type=str,help='testset,(flickr or vggss)')

    # 2. data의 root directory 경로를 지정 
    parser.add_argument('--data_path', default='',type=str,help='Root directory path of data')
    
    # 3. image size : 입력 이미지의 크기 지정
    parser.add_argument('--image_size',default=224,type=int,help='Height and width of inputs')
    
    # 4. ground truth data의 경로 (필요 시)
    parser.add_argument('--gt_path',default='',type=str)

    # 5. model 가중치나 checkpoint를 저장할 경로
    parser.add_argument('--summaries_dir',default='',type=str,help='Model path')
    
    # 6. batch size 지정
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    
    # 7. 양성 샘플에 대한 기준(threshold) 설정
    parser.add_argument('--epsilon', default=0.65, type=float, help='pos')
    
    # 8. 음성 샘플에 대한 기준(threshold) 설정
    parser.add_argument('--epsilon2', default=0.4, type=float, help='neg')
    
    # 9. --tri_map 활성화 플래그
    parser.add_argument('--tri_map',action='store_true')
    parser.set_defaults(tri_map=True)

    # 10. 음성 샘플로 처리하는 옵션 활성화하는 플래그
    parser.add_argument('--Neg',action='store_true')
    parser.set_defaults(Neg=True)

    # 이부분 내가 수정
    parser.add_argument('--random_threshold', type=float, default=0.03, help='Threshold for random sampling (if used)')


    return parser.parse_args() 

def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # load model
    model= AVENet(args) # 모델 초기화
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model) # 모델 병렬 처리
    model = model.cuda() # model을 cuda로 옮김
    checkpoint = torch.load(args.summaries_dir) # 저장된 checkpoint 불러오기
    model_dict = model.state_dict() # 모델의 현재 상태 딕셔너리
    pretrained_dict = checkpoint['model_state_dict'] # 체크포인트에서 불러온 상태 딕셔너리
    model_dict.update(pretrained_dict) # 체크포인트에서 불러온 가중치로 업데이트
    model.load_state_dict(model_dict)
    model.to(device)
    print('load pretrained model.')

    # dataloader
    testdataset = GetAudioVideoDataset(args,  mode='test') # test dataset load
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers = 16) # test data loader 생성
    print("Loaded dataloader.")

    # gt for vggss
    # if args.testset == 'vggss':
    #     # args.testset이 vggss 일 때, VGG Sound Dataset에 대한 GT(ground truth)를 load
    #     args.gt_all = {} # 이 딕셔너리에 gt 정보 저장
    #     # with open('metadata/vggss.json') as json_file:
    #     with open('metadata/test.json') as json_file:
    #         annotations = json.load(json_file)
    #     # for annotation in annotations:
    #     #     args.gt_all[annotation['file']] = annotation['bbox'] # 각 파일에 대한 bounding box 정보를 args.gt_all에 저장


    model.eval()

    # Embeddings and retrieval
    image_embeddings = []
    audio_embeddings = []
    ids = []

#     for step, (image, spec, audio,name,im) in enumerate(testdataloader):
#         # testdataloader에서 image, spec, audio, name, im 데이터를 한 배치씩 가져옴.
#         print('%d / %d' % (step,len(testdataloader) - 1))
#         spec = Variable(spec).cuda()
#         image = Variable(image).cuda()
#         heatmap,_,Pos,Neg = model(image.float(),spec.float(),args)
#         heatmap_arr =  heatmap.data.cpu().numpy()

#         for i in range(spec.shape[0]):
#             heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
#             heatmap_now = normalize_img(-heatmap_now)
#             gt_map = testset_gt(args,name[i])
#             pred = 1 - heatmap_now
#             threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
#             pred[pred>threshold]  = 1
#             pred[pred<1] = 0
#             evaluator = Evaluator()
#             ciou,inter,union = evaluator.cal_CIOU(pred,gt_map,0.5)
#             iou.append(ciou)

#     results = []
#     for i in range(21):
#         result = np.sum(np.array(iou) >= 0.05 * i)
#         result = result / len(iou)
#         results.append(result)
#     x = [0.05 * i for i in range(21)]
#     auc_ = auc(x, results)
#     print('cIoU' , np.sum(np.array(iou) >= 0.5)/len(iou))
#     print('auc',auc_)


# if __name__ == "__main__":
#     main()

    with torch.no_grad():
        for step, (image, spec, _, name, _) in enumerate(testdataloader):
            print(f'{step} / {len(testdataloader) - 1}')
            image = image.cuda().float()
            spec = spec.cuda().float()

            # Extract features for retrieval
            img_emb, aud_emb = model.extract_features(image, spec)
            image_embeddings.append(img_emb.cpu().numpy())
            audio_embeddings.append(aud_emb.cpu().numpy())
            ids.extend(name)

    # Concatenate embeddings and calculate similarity
    image_embeddings = np.vstack(image_embeddings)
    audio_embeddings = np.vstack(audio_embeddings)
    similarity_matrix = cosine_similarity(image_embeddings, audio_embeddings)

    # Evaluate retrieval performance
    retrieval_results = evaluate_retrieval(similarity_matrix, ids)
    print(f'Top-1 Retrieval Accuracy: {retrieval_results["top_1_accuracy"]:.4f}')
    print(f'Top-5 Retrieval Accuracy: {retrieval_results["top_5_accuracy"]:.4f}')
    print(f'AUC: {retrieval_results["auc"]:.4f}')

def evaluate_retrieval(similarity_matrix, ids):
    top_k = [1, 5]
    correct_top_k = {k: 0 for k in top_k}
    iou = []

    for i in range(len(ids)):
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        
        for k in top_k:
            if i in sorted_indices[:k]:
                correct_top_k[k] += 1

    top_1_accuracy = correct_top_k[1] / len(ids)
    top_5_accuracy = correct_top_k[5] / len(ids)
    auc_ = auc([0.05 * i for i in range(21)], [correct_top_k[1] / len(ids) for i in range(21)])

    return {"top_1_accuracy": top_1_accuracy, "top_5_accuracy": top_5_accuracy, "auc": auc_}

if __name__ == "__main__":
    main()
