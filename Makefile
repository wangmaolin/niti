all: cutlass pool mm conv 
	
cutlass:
	rm -rf pytorch/cutlass-extension/build pytorch/cutlass-extension/dist 
	rm -rf pytorch/cutlass-extension/cutlassconv.egg-info
	cd pytorch/cutlass-extension; python setup.py install

pool:
	rm -rf pytorch/int8pool-extension/build pytorch/int8pool-extension/dist 
	rm -rf pytorch/int8pool-extension/int8pool.egg-info
	cd pytorch/int8pool-extension; python setup.py install

mm:
	rm -rf pytorch/int8mm-extension/build pytorch/int8mm-extension/dist 
	rm -rf pytorch/int8mm-extension/int8mm.egg-info
	cd pytorch/int8mm-extension; python setup.py install

conv:
	rm -rf pytorch/int8conv-extension/build pytorch/int8conv-extension/dist 
	rm -rf pytorch/int8conv-extension/int8conv.egg-info
	cd pytorch/int8conv-extension; python setup.py install

clean:
	rm -rf pytorch/int8mm-extension/build pytorch/int8mm-extension/dist 
	rm -rf pytorch/cutlass-extension/build pytorch/cutlass-extension/dist 
	rm -rf pytorch/int8pool-extension/build pytorch/int8pool-extension/dist 
	rm -rf pytorch/int8conv-extension/build pytorch/int8conv-extension/dist 
	rm -rf pytorch/cutlass-extension/cutlassconv.egg-info
	rm -rf pytorch/int8pool-extension/int8pool.egg-info
	rm -rf pytorch/int8mm-extension/int8mm.egg-info
	rm -rf pytorch/int8conv-extension/int8conv.egg-info