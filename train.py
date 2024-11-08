import os
import argparse
import builtins
import time
import numpy as np

import torch
import json
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist

import utils
from model import AVENet
from DatasetLoader import GetAudioVideoDataset
from tqdm import tqdm

# from datasets import get_train_dataset, get_test_dataset
from torch.utils.tensorboard import SummaryWriter



import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    
    # Model and experiment configurations
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--experiment_name', type=str, default='hdvsl_vggss', help='Experiment name for checkpointing and logging')

    # Dataset paths and configurations
    parser.add_argument('--trainset', type=str, default='vggss', help='Name of the training dataset (e.g., flickr, vggss)')
    parser.add_argument('--testset', type=str, default='vggss', help='Name of the test dataset (e.g., flickr, vggss)')
    parser.add_argument('--image_size',default=224,type=int,help='Height and width of inputs')

    # Model hyperparameters as used in AVENet class
    parser.add_argument('--epsilon', type=float, default=0.65, help='Threshold for positive cases in similarity calculation')
    parser.add_argument('--epsilon2', type=float, default=0.4, help='Threshold for negative cases in similarity calculation')
    # epsilon이랑 epsilon2 필요한지 아직 잘 모르겠음. (positive, negative 하지 말라는 것 같았음)

    parser.add_argument('--tri_map', action='store_true', help='Use tri-map for additional negative cases')
    parser.set_defaults(tri_map=True)
    parser.add_argument('--Neg', action='store_true', help='Include negative samples in similarity calculation')
    parser.set_defaults(Neg=True)
    parser.add_argument('--random_threshold', type=float, default=0.03, help='Threshold for random sampling (if used)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--init_lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed for reproducibility')

    # Distributed training parameters
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of nodes for distributed training')
    parser.add_argument('--rank', type=int, default=0, help='Node rank for distributed training')
    parser.add_argument('--node', type=str, default='localhost', help='Node hostname')
    parser.add_argument('--port', type=int, default=12345, help='Port for distributed training communication')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345', help='URL for distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training')

    return parser.parse_args()


def main(args):
    mp.set_start_method('spawn') # 다중 프로세스를 실행할 때 spawn 방식을 사용하여 각 프로세스를 독립적으로 시작.
    args.dist_url = f'tcp://{args.node}:{args.port}' # 분산 학습에서 사용할 URL을 생성
    print('Using url {}'.format(args.dist_url))

    ngpus_per_node = torch.cuda.device_count() # 현재 노드에서 사용할 수 있는 GPU 개수를 파악
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    # Setup distributed environment
    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Create model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    # Create model
    model = AVENet(args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    print(model)

    # Optimizer
    optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(model, args)

    # Resume if possible
    start_epoch, best_cIoU, best_Auc = 0, 0., 0. # 초기화 단계로 처음에 변수를 0으로 설정
    if os.path.exists(os.path.join(model_dir, 'latest.pth')):
        ckp = torch.load(os.path.join(model_dir, 'latest.pth'), map_location='cpu')
        start_epoch, best_cIoU, best_Auc = ckp['epoch'], ckp['best_cIoU'], ckp['best_Auc'] # 로드한 checkpoint file에서 epoch, best_cIoU, best_Auc 값을 가져와 할당
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        print(f'loaded from {os.path.join(model_dir, "latest.pth")}')

    # Dataloaders
    # traindataset = get_train_dataset(args)
    traindataset = GetAudioVideoDataset(args, mode = 'train')
    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size= args.batch_size, shuffle = (train_sampler is None), num_workers = 16)
    
    # train_sampler = None
    # if args.multiprocessing_distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    # train_loader = torch.utils.data.DataLoader(
    #     traindataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
    #     persistent_workers=args.workers > 0)

    # testdataset = get_test_dataset(args)
    # test_loader = torch.utils.data.DataLoader(
    #     testdataset, batch_size=1, shuffle=False,
    #     num_workers=args.workers, pin_memory=False, drop_last=False,
    #     persistent_workers=args.workers > 0)

    testdataset = GetAudioVideoDataset(args,  mode='test')
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers = 1)

    print("Loaded dataloader.")

    # =============================================================== #
    # Training loop
    # cIoU, auc = validate(test_loader, model, args)
    # print(f'cIoU (epoch {start_epoch}): {cIoU}')
    # print(f'AUC (epoch {start_epoch}): {auc}')
    # print(f'best_cIoU: {best_cIoU}')
    # print(f'best_Auc: {best_Auc}')

    for epoch in range(start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train(train_loader, model, optimizer, epoch, args)

        # Evaluate
        cIoU, auc = validate(test_loader, model, args)
        print(f'cIoU (epoch {epoch+1}): {cIoU}')
        print(f'AUC (epoch {epoch+1}): {auc}')
        print(f'best_cIoU: {best_cIoU}')
        print(f'best_Auc: {best_Auc}')

        # Checkpoint
        if args.rank == 0:
            ckp = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch+1,
                   'best_cIoU': best_cIoU,
                   'best_Auc': best_Auc}
            torch.save(ckp, os.path.join(model_dir, 'latest.pth'))
            print(f"Model saved to {model_dir}")
        if cIoU >= best_cIoU:
            best_cIoU, best_Auc = cIoU, auc
            if args.rank == 0:
                torch.save(ckp, os.path.join(model_dir, 'best.pth'))


def load_labels():
    with open("/mnt/scratch/users/individuals/VGGsound_individual/metadata/test.json", 'r') as f:
        full_data = json.load(f)  # JSON 파일 전체를 로드
        data = full_data["data"]  # "data" 키를 통해 실제 데이터에 접근

    labels = {}
    for item in data:
        labels[item['video_id']] = item['labels']
    
    return labels


def train(train_loader, model, optimizer, epoch, args):
    print("train 들어옴")
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_mtr = AverageMeter('Loss', ':.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_mtr],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i, (image, spec, _, _,_) in enumerate(train_loader):
        data_time.update(time.time() - end)

        #if args.gpu is not None:
        spec = spec.cuda()
        image = image.cuda()

        image_emb, audio_emb = model.extract_features(image, spec)
        
        print("Image Emb requires_grad:", image_emb.requires_grad)
        print("Audio Emb requires_grad:", audio_emb.requires_grad)

        similarity_matrix = torch.mm(image_emb, audio_emb.T)

        print("Similarity Matrix grad_fn:", similarity_matrix.grad_fn)  

        labels = torch.arange(similarity_matrix.size(0)).long().cuda()
        loss = F.cross_entropy(similarity_matrix, labels)
        print("Loss grad_fn:", loss.grad_fn)  # None이 아니어야 함

        
        loss_mtr.update(loss.item(), image.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 or i == len(train_loader) - 1:
            progress.display(i)

        del loss


def validate(test_loader, model, args):
    model.cuda()
    model.eval()
    
    # 유사도 계산을 위한 평가 도구 초기화
    image_embeddings = []
    audio_embeddings = []
    ids = []
    evaluator = utils.Evaluator()

    # 데이터셋에서 이미지와 오디오 임베딩 추출
    for step, (image, spec, _, name, _) in tqdm(enumerate(test_loader), desc="Embedding Extraction", total=len(test_loader)):
        image, spec = image.cuda().float(), spec.cuda().float()

        with torch.no_grad():
            img_emb, aud_emb = model.extract_features(image, spec)
            
            # 이미지와 오디오 임베딩이 4차원일 경우 평균 풀링 적용 (B x 512 x W x H -> B x 512)
            if img_emb.dim() == 4:
                img_emb = F.avg_pool2d(img_emb, kernel_size=(img_emb.size(2), img_emb.size(3))).squeeze()
            if aud_emb.dim() == 4:
                aud_emb = F.avg_pool2d(aud_emb, kernel_size=(aud_emb.size(2), aud_emb.size(3))).squeeze()

        # 추출된 이미지와 오디오 임베딩 추가
        image_embeddings.append(img_emb)
        audio_embeddings.append(aud_emb)
        ids.extend(name)

    # 텐서로 결합
    image_embeddings = torch.cat(image_embeddings, dim=0)
    audio_embeddings = torch.cat(audio_embeddings, dim=0)

    # 유사도 행렬 계산 (블록 단위)
    similarity_matrix = torch.zeros(image_embeddings.size(0), audio_embeddings.size(0))
    batch_size = 128
    for i in range(0, image_embeddings.size(0), batch_size):
        for j in range(0, audio_embeddings.size(0), batch_size):
            img_batch = image_embeddings[i:i + batch_size].cuda()
            aud_batch = audio_embeddings[j:j + batch_size].cuda()
            similarity_matrix[i:i + batch_size, j:j + batch_size] = torch.mm(img_batch, aud_batch.T).cpu()

    # GT 맵 설정 (같은 label이 있으면 1로 설정)
    labels = load_labels()
    gt_map = np.zeros_like(similarity_matrix)
    for i, id1 in enumerate(ids):
        for j, id2 in enumerate(ids):
            if labels[id1] == labels[id2]:  # 같은 label이면 유사한 쌍으로 정의
                gt_map[i, j] = 1

    # cIoU 및 AUC 계산
    for i in range(similarity_matrix.shape[0]):
        # `pred`와 `gt_map[i]`을 1차원에서 224x224 크기로 변환
        pred = similarity_matrix[i].cpu().view(1, 1, 15446, 1)
        pred = F.interpolate(pred, size=(224, 224), mode='bicubic', align_corners=False).squeeze().numpy()
        
        gt_map_resized = torch.tensor(gt_map[i]).view(1, 1, 15446, 1)
        gt_map_resized = F.interpolate(gt_map_resized, size=(224, 224), mode='nearest').squeeze().numpy()

        thr = np.sort(pred.flatten())[int(pred.size / 2)]
        evaluator.cal_CIOU(pred, gt_map_resized, thr)

    # 최종 성능 평가
    cIoU = evaluator.final()
    AUC = evaluator.cal_AUC()
    return cIoU, AUC




# def validate(test_loader, model, args):
#     model.cuda()
#     model.eval()
#     image_embeddings = []
#     audio_embeddings = []
#     ids = []

#     from tqdm import tqdm
#     # 데이터셋에서 이미지와 오디오 임베딩 추출
#     for step, (image, spec, _, name, _) in tqdm(enumerate(test_loader)):
        
#         spec = spec.cuda().float()
#         image = image.cuda().float()

#         # 이미지 및 오디오 임베딩 생성
#         with torch.no_grad():
#             img_emb, aud_emb = model.extract_features(image, spec)
            
#             # 평균 풀링을 통해 B x 512 x W x H -> B x 512로 변환
#             if img_emb.dim() == 4:  # B x 512 x W x H 형태일 경우
#                 img_emb = F.avg_pool2d(img_emb, kernel_size=(img_emb.size(2), img_emb.size(3))).squeeze()
#             if aud_emb.dim() == 4:
#                 aud_emb = F.avg_pool2d(aud_emb, kernel_size=(aud_emb.size(2), aud_emb.size(3))).squeeze()

#         # B x 512 형태로 변환된 임베딩을 사용
#         image_embeddings.append(img_emb)
#         audio_embeddings.append(aud_emb)
#         ids.extend(name)

#     # 텐서로 결합
#     image_embeddings = torch.cat(image_embeddings, dim=0)
#     audio_embeddings = torch.cat(audio_embeddings, dim=0)

#     # 유사도 행렬 계산
#     similarity_matrix = torch.mm(image_embeddings, audio_embeddings.T)

#     # Retrieval 성능 평가
#     retrieval_results = evaluate_retrieval(similarity_matrix, ids)
#     print("여기까지 옴")
#     return retrieval_results


# def validate(test_loader, model, args):
#     model.cuda()
#     model.eval()
#     image_embeddings = []
#     audio_embeddings = []
#     ids = []

#     from tqdm import tqdm
#     # 데이터셋에서 이미지와 오디오 임베딩 추출
#     for step, (image, spec, _, name, _) in tqdm(enumerate(test_loader)):
        
#         spec = spec.cuda().float()
#         image = image.cuda().float()

#         # 이미지 및 오디오 임베딩 생성
#         with torch.no_grad():
#             img_emb, aud_emb = model.extract_features(image, spec)
        
#         # GPU 메모리 사용을 최소화하기 위해 CPU로 이동
#         image_embeddings.append(img_emb)
#         audio_embeddings.append(aud_emb)
#         ids.extend(name)

#     # CPU에서 텐서로 결합
#     image_embeddings = torch.cat(image_embeddings, dim=0)
#     audio_embeddings = torch.cat(audio_embeddings, dim=0)

#     # 결합된 텐서를 2D로 변환
#     image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
#     audio_embeddings = audio_embeddings.view(audio_embeddings.size(0), -1)

#     # 큰 유사도 행렬을 작은 블록으로 나눠 계산
#     similarity_matrix = []
#     batch_size = 1024  # 필요한 메모리 여유에 따라 조절
    
#     for i in range(0, image_embeddings.size(0), batch_size):
#         # 배치 크기만큼 슬라이스하여 행렬 곱셈 수행
#         img_batch = image_embeddings[i:i+batch_size]
#         sim_batch = torch.mm(img_batch, audio_embeddings.T)  # CPU에서 연산 수행
#         similarity_matrix.append(sim_batch)

#     # 유사도 행렬을 다시 결합
#     similarity_matrix = torch.cat(similarity_matrix, dim=0)

#     # Retrieval 성능 평가
#     retrieval_results = evaluate_retrieval(similarity_matrix, ids)
#     print("여기까지 옴")
#     return retrieval_results



def evaluate_retrieval(similarity_matrix, ids):
    top_k = 5  # Top-K 정확도 평가 예시
    correct_top_k = 0

    for i in range(len(ids)):
        sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
        top_k_indices = sorted_indices[:top_k]

        if i in top_k_indices:
            correct_top_k += 1

    accuracy_top_k = correct_top_k / len(ids)
    print(f'Top-{top_k} Retrieval Accuracy: {accuracy_top_k:.4f}')
    return accuracy_top_k



class AverageMeter(object):
    # 평균값을 계산하고 저장하는 도우미 class
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main(get_arguments())
