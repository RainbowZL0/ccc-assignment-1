import pprint

from a004_assignment_1.a000_CFG import (
    RAW_DATA_FOLDER, NDJSON_FILE_NAME_TO_LOAD,
    FILTERED_DATA_FOLDER,
    SIZE,
    RANK,
    TEST_DATA_FOLDER,
    COMM, NDJSON_LINE_NUM,
)
from a004_assignment_1.a002_utils import (
    write_data_to_ndjson,
    load_ndjson_file,
    aggregate_score_by_hour,
    split_list,
    join_dict_pieces_hour_score,
    load_ndjson_file_by_process,
    measure_time,
)


def start_one_core():
    records: list = load_ndjson_file(
        ndjson_path_for_loading=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
        use_filter=True,
    )
    write_data_to_ndjson(
        records=records, target_path=FILTERED_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    )
    pprint.pprint(aggregate_score_by_hour(records))


def mpi_v1():
    """主进程读取全部数据，然后分发给工作进程"""
    if RANK == 0:
        records: list | None = load_ndjson_file(
            ndjson_path_for_loading=RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
            use_filter=True,
        )
        print(f"1. rank={RANK}, 节点读取数据结束")

        records = split_list(lst=records, pieces_num=SIZE)
        print(f"2. rank={RANK}, 数据分片结束")

        # received_msg = comm.scatter(records, root=0)
        # hour_score: dict = aggregate_score_by_hour(received_msg)

        # target_name = get_ndjson_name_by_rank(
        #     original_name=NDJSON_FILE_NAME_TO_LOAD,
        #     r=rank,
        # )
        # target_path = TEST_DATA_FOLDER / target_name
        # write_data_to_ndjson(
        #     records=hour_score,
        #     target_path=target_path
        # )

        # gathered_msg = comm.gather(hour_score, root=0)
    else:
        records = None
        # received_msg = comm.scatter(None, root=0)
        # hour_score = aggregate_score_by_hour(received_msg)
        #
        # target_name = get_ndjson_name_by_rank(
        #     original_name=NDJSON_FILE_NAME_TO_LOAD,
        #     r=rank,
        # )
        # target_path = TEST_DATA_FOLDER / target_name
        # write_data_to_ndjson(
        #     records=hour_score,
        #     target_path=target_path
        # )
        # _ = comm.gather(hour_score, root=0)

    received_msg = COMM.scatter(records, root=0)
    print("3. rank={RANK}, 分发数据结束")
    
    hour_score = aggregate_score_by_hour(received_msg)
    print("4. rank={RANK}, 节点统计数据结束")
    
    all_hour_score = COMM.gather(hour_score, root=0)
    print("5. rank={RANK}, gather结束")

    if RANK == 0:
        all_hour_score = join_dict_pieces_hour_score(all_hour_score, mode="sum")
        print("6. rank={RANK}, 汇总结束")
        
        write_data_to_ndjson(
            records=all_hour_score,
            target_path=TEST_DATA_FOLDER / "gathered.ndjson"
        )
        print(f"7. rank={RANK}, 保存结果到磁盘结束")


def mpi_v2():
    """所有进程同时开始读取数据"""
    ndjson_path = RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    ndjson_line_num = NDJSON_LINE_NUM

    records = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=SIZE,
        r=RANK,
        use_filter=True,
    )
    print(f"1. rank={RANK}, 节点读取数据结束")

    hour_score = aggregate_score_by_hour(records)
    print(f"2. rank={RANK}, 节点分别统计结束")
    
    all_hour_score = COMM.gather(hour_score, root=0)
    print(f"3. rank={RANK}, gather结束")

    if RANK == 0:
        all_hour_score = join_dict_pieces_hour_score(all_hour_score, mode="sum")
        print(f"4. rank={RANK}, 在主节点汇总结束")
        
        write_data_to_ndjson(
            records=all_hour_score,
            target_path=TEST_DATA_FOLDER / "gathered_v2.ndjson"
        )
        print(f"5. rank={RANK}, 保存结果到磁盘结束")


def measure_mpi_v1():
    if RANK == 0:
        print(measure_time(mpi_v1)())
    else:
        mpi_v1()


def measure_mpi_v2():
    if RANK == 0:
        print(measure_time(mpi_v2)())
    else:
        mpi_v2()


# noinspection PyUnusedLocal
def tst_load_ndjson_file_by_process():
    ndjson_path = RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    ndjson_line_num = NDJSON_LINE_NUM

    records_0 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=0,
        use_filter=True,
    )
    records_1 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=1,
        use_filter=True,
    )
    records_2 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=2,
        use_filter=True,
    )
    records_3 = load_ndjson_file_by_process(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=4,
        r=3,
        use_filter=True,
    )
    pass


if __name__ == "__main__":
    # mpi_v1()
    # tst_load_ndjson_file_by_process()
    measure_mpi_v1()
