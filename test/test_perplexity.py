import torch
from torch.autograd import Variable
import torch.nn.functional as F

from main import corpus, eval_cross_entropy, calc_cross_entropy


MOCK_OUTPUT = torch.Tensor(
    [[1, 2, 0],
     [-1, -2, 0],
     [0.5, 0.2, -0.3]]
)

MOCK_LABEL = torch.LongTensor(
    [1, 2, 1]
).unsqueeze(0)

MOCK_LENGTH = torch.LongTensor(
    [2]
)

EXPECT_LOSS = 0.8


def test_import():
    assert torch
    assert corpus


def around(source, target):
    return abs(source - target) <= 0.2


def test_calc_cross_entropy():
    def softmax(tensor):
        v = Variable(tensor)
        output = F.log_softmax(v).data
        return output
    MOCK_OUTPUT_LOG_SOFTMAX = softmax(MOCK_OUTPUT).unsqueeze(0)
    print(MOCK_OUTPUT_LOG_SOFTMAX)
    loss = calc_cross_entropy(
        MOCK_OUTPUT_LOG_SOFTMAX, MOCK_LABEL, MOCK_LENGTH)
    assert around(loss, EXPECT_LOSS)


def test_eval_cross_entropy():
    loss = eval_cross_entropy(
        Variable(MOCK_OUTPUT)[:MOCK_LENGTH[0]].unsqueeze(0),
        Variable(MOCK_LABEL)[:, :MOCK_LENGTH[0]],
        MOCK_LENGTH
    )
    assert around(loss, EXPECT_LOSS)

def test_batch_eval_cross_entropy():
    BATCH_MOCK_OUTPUT = torch.stack([MOCK_OUTPUT, MOCK_OUTPUT])
    BATCH_MOCK_LABEL = torch.stack([MOCK_LABEL, MOCK_LABEL])
    BATCH_MOCK_LENGTH = torch.LongTensor(
        [3, 2]
    )

    BATCH_MOCK_OUTPUT = Variable(BATCH_MOCK_OUTPUT)
    BATCH_MOCK_LABEL = Variable(BATCH_MOCK_LABEL)

    loss = eval_cross_entropy(BATCH_MOCK_OUTPUT, BATCH_MOCK_LABEL, BATCH_MOCK_LENGTH)

    assert around(loss / 5, 0.5)
