import torch.nn as nn

# from config.config import load_opts
# opts = load_opts()

# if opts.norm == 'bn':
norm1d = nn.BatchNorm1d
norm2d = nn.BatchNorm2d
norm3d = nn.BatchNorm3d
# elif opts.norm == 'group':
#     norm1d = norm2d = norm3d = lambda num_features: nn.GroupNorm( num_groups = num_features//8, num_channels = num_features )

class DebugModule(nn.Module):
    """
    Wrapper class for printing the activation dimensions and memory usage per layer
    """
    def __init__(self, name=None):
        super().__init__()
        self.debug_log = [1] # 활성화 여부를 결정하는 리스트
        self.totmem = 0 # 메모리 사용량을 누적하는 변수
        self.name = name # 디버깅 메시지에 포함될 선택적인 이름

    def debug_line(self, layer_str, output, memuse=1, final_call = False):
        if self.debug_log[0]: # debug_log[0] == 1일때만 실행
            import numpy as np
            mem = np.prod(output.shape) * 4 # 4 bytes for float32
            # output.shape의 원소 수를 기반으로 메모리 사용량을 계산
            self.totmem += mem # 최종적으로 누적된 메모리 사용량 totmem
            memstr = ' Memory usage: {:,} Bytes'.format(mem) if memuse else ''
            namestr = '{}: '.format(self.name) if self.name is not None else ''
            print('{}{:80s}: dims {}{}'.format( namestr, repr(layer_str), output.shape, memstr))

            if final_call:
                self.debug_log[0] = 0
                print('Total memory usage: {:,} Bytes'.format(self.totmem) )
                print()