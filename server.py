import threading
import torch
from torch import nn
from torch import distributed

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(5, 5)
        self.emb = nn.Embedding(5, 5)

    def forward(self, input):
        return self.linear(self.emb(input))

target_value = torch.Tensor(
    [0.1, 0.2, -0.4, 0.1, 1]
)

fake_data = torch.LongTensor(
    [
        [1, 2, 3, 4, 0],
        [3, 2, 1, 4, 0],
        [1, 0, 0, 0, 0],
        [4, 2, 3, 4, 4],
    ]
)

total_iter = 200000

distributed.init_process_group(backend='mpi', world_size=0, rank=0)
net = Net()
rank = distributed.get_rank()
world_size = distributed.get_world_size()
optimizer = torch.optim.SGD(lr=0.001, params=net.parameters(), weight_decay=1e-6, momentum=0.3)
if rank == 0:
    print('initialized, server')

    def recv_grad(sender_rank):
        counter = 0
        for _ in range(total_iter):
            for p in net.parameters():
                tensor_buffer = p.new(p.size())
                distributed.recv(tensor_buffer, sender_rank)
                if p.grad is not None:
                    p.grad.data += tensor_buffer
                else:
                    p.grad = tensor_buffer
            optimizer.step()
            counter += 1
            if counter == 100:
                counter = 0
                print('syncing parameters for {}'.format(sender_rank))
                for p in net.parameters():
                    distributed.send(p.data, sender_rank)
    threads = [threading.Thread(target=recv_grad, args=(r+1,)) for r in range(world_size - 1)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print('server: haha')
else:
    from torch.nn.utils import clip_grad_norm
    print('initialized, client')
    counter = 0
    try:
        for _ in range(total_iter):
            res = net(fake_data) - target_value
            res = res.norm(2)
            optimizer.zero_grad()
            res.backward()
            clip_grad_norm(net.parameters(), 0.15)
            for p in net.parameters():
                # p.grad.data.zero_()
                distributed.send(p.grad.data, 0)

            optimizer.step()

            counter += 1
            if counter == 100:
                print(p.norm(2))
                counter = 0
                for p in net.parameters():
                    tensor_buffer = p.new(p.size())
                    distributed.recv(tensor_buffer, 0)
                    p.data.set_(tensor_buffer)
    except KeyboardInterrupt:
        print('client: hehe')
