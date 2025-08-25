import argparse
import os
import pandas as pd

def main(datadir):
    test_path = os.path.join(datadir, "test.txt")
    print(f"Processing {test_path}")
    
    test = pd.read_csv(test_path, sep="\t", names=["h", "r", "t"])
    test_pred = pd.concat([
        test.drop_duplicates(subset=['r', 'h']),
        test.drop_duplicates(subset=['r', 't'])
    ]).drop_duplicates()
    
    output_path = os.path.join(datadir, "test_pred.txt")
    test_pred.to_csv(output_path, sep="\t", header=False, index=False)
    print(f"test_pred.txt written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test_pred.txt containing minimal \
        number of triplets from test.txt to obtain predictions")
    parser.add_argument('--datadir', required=True, help='Directory containing the test.txt file')
    args = parser.parse_args()
    main(args.datadir)