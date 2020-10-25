all: clean install

install:
	cd pytorch/int8im2col-extension; python3 setup.py install
	cd pytorch/tcint8mm-extension; python3 setup.py install

clean: 
	rm -rf pytorch/int8im2col-extension/build pytorch/int8im2col-extension/dist pytorch/int8im2col-extension/int_im2col.egg-info
	rm -rf pytorch/tcint8mm-extension/build pytorch/tcint8mm-extension/dist pytorch/tcint8mm-extension/int8mm.egg-info
