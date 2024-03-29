import os
import torch.optim as optim
from DLDL import ipDataset, ipCNN, loss, split
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter


def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',  # 'nccl' is recommended for GPUs, 'gloo' for CPUs
        init_method='env://',  # Assumes environment variables are set by the job script
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)  # Assign a GPU to each process


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, data_path, labels_path, prog_dir, jobID,\
        num_epochs = 100, log_interval = 20):
    setup(rank, world_size)

    # Make sure each process has a different seed if you are using any randomness
    torch.manual_seed(42 + rank)

    dataset = ipDataset(data_path, labels_path)
    train, dev, test = split(dataset)

    # Use DistributedSampler
    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train, batch_size=128, sampler=train_sampler, pin_memory=True)

    model = ipCNN().cuda(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if rank == 0:
        writer = SummaryWriter(prog_dir+jobID)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(rank), target.cuda(rank)
            optimizer.zero_grad()
            output = model(data)
            L = loss(output, target)
            L.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, "+\
                      f"[{batch_idx * len(data)}/{len(train_loader.dataset)}] "+\
                      f"Loss {L.item()}")
                if rank == 0:
                    writer.add_scalar('Training Loss', L.item(),\
                                      epoch*len(train_loader.dataset) + batch_idx)


        # Save model - ensure only one orocess does this or save in each process with unique filenames
        if rank == 0:
            torch.save(model.state_dict(), prog_dir+jobID+"_params_e"+\
                      str(epoch)+".pt")
    if rank == 0:
        writer.close()

    cleanup()


if __name__ == "__main__":
    data_path = '/eagle/fusiondl_aesp/jrodriguez/processed_data/processed_dataset.pt'
    labels_path = '/eagle/fusiondl_aesp/jrodriguez/processed_data/processed_labels.pt'
    prog_dir = '/eagle/fusiondl_aesp/jrodriguez/train_progress/'

    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))  # Default to 1 if not set
    train(rank, world_size, data_path, labels_path, prog_dir,\
            jobID = "DLDL_test", num_epochs = 100, log_interval = 20)