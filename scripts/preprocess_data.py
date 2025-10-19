import argparse
import polars as pl

def visualize_data(file_path):
    df = pl.read_parquet(file_path)
    print(df.head())
    print(df.describe())
    print(df.columns)

    import matplotlib.pyplot as plt
    df.select(['Date', "('Close', 'AAPL')"]).to_pandas().plot(x='Date', y="('Close', 'AAPL')")
    plt.title('Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to the Parquet file to visualize.')
    args = parser.parse_args()
    visualize_data(args.file)

if __name__ == '__main__':
    main()