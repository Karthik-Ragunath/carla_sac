import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
import time
EPISODE_LENGTH = 5000
class DrivingModel(nn.Module):
    def __init__(self):
        super(DrivingModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = DrivingModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    for i in range(EPISODE_LENGTH):
        outputs = ddp_model(torch.randn(20,10))
        labels = torch.randn(20, 5).to(device_id)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        print("Running on rank " + str(rank)+" :"+str(labels))

if __name__ == "__main__":
    start_time = time.time()
    demo_basic()
    end_time = time.time()
    print("Total Time: ", end_time - start_time)
