import os
import csv

# Read all tickers from the data folder
data_folder = './data'
tickers = [f.split('.')[0] for f in os.listdir(data_folder) if f.endswith('.csv') and f != 'last_updated.csv']

# Read indicators from indicators.csv
indicators = {}
with open('indicators.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        indicators[row[0]] = (int(row[1]), int(row[2]))

# Generate branches
branches = []
for indicator, (begin, end) in indicators.items():
    for indicator_ticker in tickers:
        for period in range(begin, end + 1, 5):
            for threshold in range(1, 101, 5):
                for days in [1, 3, 5, 7]:
                    for direction in ['<', '>']:
                        for trading_ticker in tickers:
                            branch = f"{indicator.lower()}-{period}-{indicator_ticker}-{direction}-{threshold}-{days}-{trading_ticker}"
                            branches.append(branch)

# Save branches to txt file
with open('branches.txt', 'w') as f:
    f.write('\n'.join(branches))

print(f"Generated {len(branches)} branches and saved to branches.txt")
