DEPTH=13
CUDA_VISIBLE_DEVICES=0\
	./ti_main.py --model-type int\
	--dataset cifar10\
	--model vgg\
    --depth $DEPTH\
	--data-dir  /media/ssd0 \
	--results-dir ./results --save cifar10-int8-vgg$DEPTH\
	--epochs 150\
	--batch-size 128\
	-j 8\
	--log-interval 50\
	--weight-decay\
 	--init ./cifar10_vgg"$DEPTH"_rebalance_init.pth.tar\