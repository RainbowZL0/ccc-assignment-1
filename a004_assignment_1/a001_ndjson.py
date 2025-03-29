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
    measure_time, load_ndjson_file_by_process_and_calcu_score_at_the_same_time,
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
    else:
        records = None

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
    # write_data_to_ndjson(
    #     records=records,
    #     target_path=FILTERED_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD,
    # )
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


def mpi_v3():
    """所有进程同时开始读取数据，且读取过程中就计算分数"""
    ndjson_path = RAW_DATA_FOLDER / NDJSON_FILE_NAME_TO_LOAD
    ndjson_line_num = NDJSON_LINE_NUM

    hour_score = load_ndjson_file_by_process_and_calcu_score_at_the_same_time(
        ndjson_path_for_loading=ndjson_path,
        ndjson_line_num=ndjson_line_num,
        process_num=SIZE,
        r=RANK,
        use_filter=False,
    )
    print(f"1. rank={RANK}, 节点读取数据并统计，结束")

    all_hour_score = COMM.gather(hour_score, root=0)
    print(f"2. rank={RANK}, gather结束")

    if RANK == 0:
        all_hour_score = join_dict_pieces_hour_score(all_hour_score, mode="sum")
        print(f"3. rank={RANK}, 在主节点统计结束")

        write_data_to_ndjson(
            records=all_hour_score,
            target_path=TEST_DATA_FOLDER / "gathered_v3.ndjson"
        )
        print(f"4. rank={RANK}, 保存结果到磁盘结束")


def measure_mpi(func):
    if RANK == 0:
        print(measure_time(func)())
    else:
        func()


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
    measure_mpi(mpi_v3)
