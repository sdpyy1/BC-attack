python main_fed.py  --attack batnet --defence multikrum --trigger square --dataset cifar --model resnet --iid 1 --malicious 0
python main_fed.py  --attack batnet --defence RLR --trigger square --dataset cifar --model resnet --iid 1 --malicious 0
python main_fed.py  --attack batnet --defence flame --trigger square --dataset cifar --model resnet --iid 1 --malicious 0
python main_fed.py  --attack batnet --defence fltrust --trigger square --dataset cifar --model resnet --iid 1 --malicious 0
python main_fed.py  --attack batnet --defence medium --trigger square --dataset cifar --model resnet --iid 1 --malicious 0
python main_fed.py  --attack batnet --defence avg --trigger square --dataset cifar --model resnet --iid 1 --malicious 0

python main_fed.py  --attack opt --defence multikrum --trigger opt --dataset cifar --model resnet --iid 1
python main_fed.py  --attack opt --defence RLR --trigger opt --dataset cifar --model resnet --iid 1
python main_fed.py  --attack opt --defence flame --trigger opt --dataset cifar --model resnet --iid 1
python main_fed.py  --attack opt --defence fltrust --trigger opt --dataset cifar --model resnet --iid 1
python main_fed.py  --attack opt --defence medium --trigger opt --dataset cifar --model resnet --iid 1
python main_fed.py  --attack opt --defence avg --trigger opt --dataset cifar --model resnet --iid 1