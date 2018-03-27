nohup python -u data_processing.py -a --analysis_files ./raw/trainset/description.search.train.json --result_dir ./raw/results/ >log.txt 2>&1 &
nohup python -u data_processing.py -a --analysis_files ./raw/trainset/description.zhidao.train.json --result_dir ./raw/results/ >log.txt 2>&1 &
nohup python -u data_processing.py -a --analysis_files ./raw/trainset/entity.search.train.json --result_dir ./raw/results/ >log.txt 2>&1 &
nohup python -u data_processing.py -a --analysis_files ./raw/trainset/entity.zhidao.train.json --result_dir ./raw/results/ >log.txt 2>&1 &
nohup python -u data_processing.py -a --analysis_files ./raw/trainset/yesno.search.train.json --result_dir ./raw/results/ >log.txt 2>&1 &
nohup python -u data_processing.py -a --analysis_files ./raw/trainset/yesno.zhidao.train.json --result_dir ./raw/results/ >log.txt 2>&1 &