from pathlib import Path

from mpi4py import MPI

DATA_FOLDER = Path("a003_data")
RAW_DATA_FOLDER = DATA_FOLDER / "a001_raw"
FILTERED_DATA_FOLDER = DATA_FOLDER / "a002_filtered"
TEST_DATA_FOLDER = DATA_FOLDER / "a003_test"

NDJSON_FILE_NAME_TO_LOAD = r"mastodon-144g.ndjson"
line_num_info = {
    "mastodon-106k.ndjson": 30,
    "mastodon-16m.ndjson": 4500,
    "mastodon-144g.ndjson": int(42e6),
}
NDJSON_LINE_NUM = line_num_info[NDJSON_FILE_NAME_TO_LOAD]

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

if __name__ == '__main__':
    print("haha")
