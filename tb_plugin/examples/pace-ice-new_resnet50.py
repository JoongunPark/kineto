#!/usr/bin/env python3

import os

import torch
torch.hub.set_dir('/storage/ice1/3/0/jtong45/Joongun/tmp')
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch._C._profiler import _ExperimentalConfig, _ExtraFields_PyCall
from torch.autograd.profiler import KinetoStepTracker, profile as _profile
from torch.autograd.profiler_legacy import profile as _profile_legacy
from torch.profiler import (
    _utils,
    DeviceType,
    ExecutionTraceObserver,
    kineto_available,
    profile,
    ProfilerAction,
    ProfilerActivity,
    record_function,
    supported_activities,
)
from torch.profiler._pattern_matcher import (
    Conv2dBiasFollowedByBatchNorm2dPattern,
    ExtraCUDACopyPattern,
    ForLoopIndexingPattern,
    FP32MatMulPattern,
    GradNotSetToNonePattern,
    MatMulDimInFP16Pattern,
    NamePattern,
    OptimizerSingleTensorPattern,
    Pattern,
    report_all_anti_patterns,
    SynchronizedDataLoaderPattern,
)

## Required modules for ExecutionTraceObserver
#from typing import Any, List, Optional
#from datetime import datetime
#from torch.profiler import (
#    _utils,
#    DeviceType,
#    ExecutionTraceObserver,
#    kineto_available,
#    profile,
#    ProfilerAction,
#    ProfilerActivity,
#    record_function,
#    supported_activities,
#)
#from time import perf_counter_ns as pc
#from torch.autograd.profiler import profile , _ExperimentalConfig
#from torchvision import models

def example(rank, use_gpu=True):
    # Register Execution Trace Observer
    eg_file = "./result/host_" + str(rank) + ".json"

    # Define global variable for custom trace_handler
    global g_rank
    g_rank = rank

    if use_gpu:
        torch.cuda.set_device(rank)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(rank)
        model.cuda()
        cudnn.benchmark = True
        model = DDP(model, device_ids=[rank])
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = DDP(model)

    # Use gradient compression to reduce communication
    # model.register_comm_hook(None, default.fp16_compress_hook)
    # or
    # state = powerSGD_hook.PowerSGDState(process_group=None,matrix_approximation_rank=1,start_powerSGD_iter=2)
    # model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler,
                                              shuffle=False, num_workers=4)

    if use_gpu:
        criterion = nn.CrossEntropyLoss().to(rank)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    print(supported_activities())
    with torch.profiler.profile(
        activities=supported_activities(),
        schedule=torch.profiler.schedule(
#            skip_first=3, wait=3, warmup=3, active=1, repeat=1
            wait=0, warmup=0, active=1, repeat=1
        ),
        with_stack=False,
        execution_trace_observer=(
            ExecutionTraceObserver().register_callback(eg_file)
        ),
#        experimental_config=_ExperimentalConfig(enable_cuda_sync_events=True),
#        record_shapes=True
    ) as p:
        for step, data in enumerate(trainloader, 0):
            print("step:{}".format(step))
            if use_gpu:
                inputs, labels = data[0].to(rank), data[1].to(rank)
            else:
                inputs, labels = data[0], data[1]

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            p.step()

            # Changed termination condition
            #if step + 1 >= 10:
            if step + 1 >= 1:
                break
        kineto_file = "./result/device_"+str(g_rank)+".json"
        p.export_chrome_trace(kineto_file)


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 8
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, example))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
