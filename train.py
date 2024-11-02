import os
import argparse
import builtins
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist

import utils
from model import AVENet
from datasets import get_train_dataset, get_test_dataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='hdvsl_vggss', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--trainset', default='vggss', type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='vggss', type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--train_data_path', default='', type=str, help='Root directory path of train data')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of test data')
    parser.add_argument('--test_gt_path', default='', type=str)

    # hd -vsl hyper-params
    # parser.add_argument('--out_dim', default=512, type=int)
    # parser.add_argument('--tau', default=0.03, type=float, help='tau')

    # training/evaluation parameters
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

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
    traindataset = get_train_dataset(args)
    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
        persistent_workers=args.workers > 0)

    testdataset = get_test_dataset(args)
    test_loader = torch.utils.data.DataLoader(
        testdataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False,
        persistent_workers=args.workers > 0)
    print("Loaded dataloader.")

    # =============================================================== #
    # Training loop
    cIoU, auc = validate(test_loader, model, args)
    print(f'cIoU (epoch {start_epoch}): {cIoU}')
    print(f'AUC (epoch {start_epoch}): {auc}')
    print(f'best_cIoU: {best_cIoU}')
    print(f'best_Auc: {best_Auc}')

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


def train(train_loader, model, optimizer, epoch, args):
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
    for i, (image, spec, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        image_emb, audio_emb = model.extract_features(image, spec)
        similarity_matrix = torch.mm(image_emb, audio_emb.T)

        labels = torch.arange(similarity_matrix.size(0)).long().cuda(args.gpu)
        loss = F.cross_entropy(similarity_matrix, labels)
        
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
    model.eval()
    image_embeddings = []
    audio_embeddings = []
    ids = []

    # 데이터셋에서 이미지와 오디오 임베딩 추출
    for step, (image, spec, _, name, _) in enumerate(test_loader):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        # 이미지 및 오디오 임베딩 생성
        with torch.no_grad():
            img_emb, aud_emb = model.extract_features(image, spec)
        
        image_embeddings.append(img_emb.cpu())
        audio_embeddings.append(aud_emb.cpu())
        ids.extend(name)

    # 임베딩 결합
    image_embeddings = torch.cat(image_embeddings, dim=0)
    audio_embeddings = torch.cat(audio_embeddings, dim=0)

    # 유사도 행렬 계산
    similarity_matrix = torch.mm(image_embeddings, audio_embeddings.T)

    # Retrieval 성능 평가
    retrieval_results = evaluate_retrieval(similarity_matrix, ids)
    return retrieval_results


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