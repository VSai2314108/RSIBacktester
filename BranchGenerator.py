import os
import csv

# Read all tickers from the data folder
data_folder = './data'
tickers_ind = []
tickers_trade = []
tickers_with_data = set()
with open('tickers-ind.txt', 'r') as file:
    tickers_ind.extend([line.strip() for line in file])
    
with open('tickers-trade.txt', 'r') as file:
    tickers_trade.extend([line.strip() for line in file])
    
data_files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and f != 'last_updated.csv']
for file in data_files:
    ticker = file.split('.')[0]
    tickers_with_data.add(ticker.upper())
    
print(tickers_ind)
print(tickers_trade)
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
    for indicator_ticker in tickers_ind:
        for period in range(begin, end + 1, 1):
            for threshold in list(range(1,30))+list(range(75,100)):
                for days in [1]:
                    for direction in ['<', '>']:
                        for trading_ticker in tickers_trade:
                            # check if we have data for both
                            if indicator_ticker in tickers_with_data and trading_ticker in tickers_with_data:
                                branch = f"{indicator.lower()}-{period}-{indicator_ticker}-{direction}-{threshold}-{days}-{trading_ticker}"
                                branches.append(branch)

# Save branches to txt file
# remove the file if it exists
if os.path.exists('branches.txt'):
    os.remove('branches.txt')

with open('branches.txt', 'w') as f:
    f.write('\n'.join(branches))

print(f"Generated {len(branches)} branches and saved to branches.txt")
