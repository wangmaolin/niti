# pytorch version tested on RTX 2080 Ti, RTX 8000  & T4
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# if you want to use RTX 3090, please use the following pytorch version
# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN pip install bokeh tensorboard pandas jupyterlab scikit-learn
WORKDIR /niti
CMD /bin/bash