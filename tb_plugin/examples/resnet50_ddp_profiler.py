import os

import torch
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

# Required modules for ExecutionTraceObserver
from typing import Any, List, Optional
from torch.profiler import ExecutionTraceObserver
from datetime import datetime
from time import perf_counter_ns as pc
from torch.autograd.profiler import profile , _ExperimentalConfig
from torchvision import models


# Custom Trace handler for Kineto Trace
def trace_handler(prof)-> Any:
    kineto_file = "kineto.rank_"+str(g_rank)+"_step_"+str(prof.step_num)
    torch.profiler.tensorboard_trace_handler("./result",worker_name=kineto_file).__call__(prof)

def example(rank, use_gpu=True):
    # Register Execution Trace Observer
    eg_file = "./result/eg.rank_" + str(rank) + ".pt.trace.json"
    eg = ExecutionTraceObserver()
    eg.register_callback(eg_file)

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

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=3,
            active=1),
        with_stack=False,
        on_trace_ready=trace_handler, # Use our custom trace handler
        experimental_config=_ExperimentalConfig(enable_cuda_sync_events=True),
        record_shapes=True
    ) as p:
        for step, data in enumerate(trainloader, 0):
            print("step:{}".format(step))
            if use_gpu:
                inputs, labels = data[0].to(rank), data[1].to(rank)
            else:
                inputs, labels = data[0], data[1]

            # When it is the active stage
            if step == 3:
                # Stop recording the trace
                eg.stop()
                eg.unregister_callback()

            # When it is the active stage
            if step == 2:
                # Record the trace
                eg.start()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            p.step()

            # Changed termination condition
            if step + 1 >= 6:
                break


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, example))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
