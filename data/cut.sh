nohup python -u data_processing.py --cut --target_files ./raw/trainset/search.train.json --size 1000 >train_search_cutlog.txt 2>&1 &
nohup python -u data_processing.py --cut --target_files ./raw/trainset/zhidao.train.json --size 1000 >train_zhidao_cutlog.txt 2>&1 &
nohup python -u data_processing.py --cut --target_files ./raw/devset/search.dev.json --size 200 >dev_search_cutlog.txt 2>&1 &
nohup python -u data_processing.py --cut --target_files ./raw/devset/zhidao.dev.json --size 200 >dev_zhidao_cutlog.txt 2>&1 &
nohup python -u data_processing.py --cut --target_files ./raw/testset/search.test.json --size 100 >test_search_cutlog.txt 2>&1 &
nohup python -u data_processing.py --cut --target_files ./raw/testset/zhidao.test.json --size 100 >test_zhidao_cutlog.txt 2>&1 &