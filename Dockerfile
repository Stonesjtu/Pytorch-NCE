FROM zhoumingjun/pytorch:latest-cuda8-py3

ADD . /lstm
WORKDIR /lstm

CMD python main.py --cuda --clip=1 --emsize 300
