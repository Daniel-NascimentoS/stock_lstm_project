import argparse
import yfinance as yf
import pandas as pd
import os

def download_stock_data(symbol, start, end, interval):
    data = yf.download(symbol, start=start, end=end, interval=interval)
    return data

def save_parquet_by_year(data, out_dir):
    data['Year'] = data.index.year
    for year, group in data.groupby('Year'):
        file_path = os.path.join(out_dir, f'{year}.parquet')
        group.drop(columns=['Year']).to_parquet(file_path)

def main():
    parser = argparse.ArgumentParser(description='Download stock data and save as yearly Parquet files.')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to download data for.')
    parser.add_argument('--start', type=str, required=True, help='Start date for data download (YYYY-MM-DD).')
    parser.add_argument('--end', type=str, required=True, help='End date for data download (YYYY-MM-DD).')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d, 1h).')
    parser.add_argument('--out_dir', type=str, default="data/raw", help='Output directory to save Parquet files.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = download_stock_data(args.symbol, args.start, args.end, args.interval)
    save_parquet_by_year(data, args.out_dir)

if __name__ == '__main__':
    main()