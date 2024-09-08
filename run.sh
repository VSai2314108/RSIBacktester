cd data
rm -rf *
cd ..
cd indicator_data
rm -rf *
cd ..

python3 DataUpdater.py
python3 PopulateIndicators.py
python3 BranchGenerator.py
python3 BranchEvaluatorPandasSparkCompat.py
python3 ResultCreatorPandasSparkCompat.py
