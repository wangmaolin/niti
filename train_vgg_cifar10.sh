DEPTH=11
    CUDA_VISIBLE_DEVICES=0\
	python -u ti_main.py --model-type int\
	--dataset cifar10\
	--model vgg\
    --depth $DEPTH\
	--data-dir /niti\
	--results-dir ./results --save test\
	--epochs 150\
	--batch-size 128\
	-j 8\
	--log-interval 50\
	--weight-decay\
 	--init /niti/cifar10_vgg"$DEPTH"_rebalance_init.pth.tar\
    --download\
