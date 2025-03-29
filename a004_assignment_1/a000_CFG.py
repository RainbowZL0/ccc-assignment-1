from pathlib import Path

from mpi4py import MPI

DATA_FOLDER = Path("a003_data")
RAW_DATA_FOLDER = DATA_FOLDER / "a001_raw"
FILTERED_DATA_FOLDER = DATA_FOLDER / "a002_filtered"
TEST_DATA_FOLDER = DATA_FOLDER / "a003_test"

NDJSON_FILE_NAME_LIST = [
    "mastodon-106k.ndjson",
    "mastodon-16m.ndjson",
    "mastodon-144g.ndjson",
]
LINE_NUM_INFO = {
    NDJSON_FILE_NAME_LIST[0]: 30,
    NDJSON_FILE_NAME_LIST[1]: 4500,
    NDJSON_FILE_NAME_LIST[2]: int(42e6),
}

NDJSON_FILE_NAME_TO_LOAD = NDJSON_FILE_NAME_LIST[1]
NDJSON_LINE_NUM = LINE_NUM_INFO[NDJSON_FILE_NAME_TO_LOAD]

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

if __name__ == '__main__':
    print("haha")
