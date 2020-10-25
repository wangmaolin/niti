FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN pip install bokeh tensorboard pandas

WORKDIR /niti

CMD /bin/bash
