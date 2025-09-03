prepare:
	mkdir -p data/raw data/interim data/processed models/artifacts models/reports

train:
	python -m src.train_model

simulate:
	python -m src.simulate
