nohup python -u data_processing.py --pre_base --seg_dir ./seg/search/train/ --target_files ./raw/trainset/search.train.json >search_train_seglog.txt 2>&1 &
nohup python -u data_processing.py --pre_base --seg_dir ./seg/zhidao/train/ --target_files ./raw/trainset/zhidao.train.json >zhidao_train_seglog.txt 2>&1 &
nohup python -u data_processing.py --pre_base --seg_dir ./seg/search/test/ --target_files ./raw/testset/search.test.json >search_test_seglog.txt 2>&1 &
nohup python -u data_processing.py --pre_base --seg_dir ./seg/zhidao/test/ --target_files ./raw/testset/zhidao.test.json >zhidao_test_seglog.txt 2>&1 &
nohup python -u data_processing.py --pre_base --seg_dir ./seg/zhidao/dev/ --target_files ./raw/devset/zhidao.dev.json >zhidao_dev_seglog.txt 2>&1 &
nohup python -u data_processing.py --pre_base --seg_dir ./seg/search/dev/ --target_files ./raw/devset/search.dev.json >search_dev_seglog.txt 2>&1 &
