import heapq
import os
import pprint

from a004_assignment_1.a000_CFG import TEST_DATA_FOLDER
from a004_assignment_1.a002_utils import parse_one_line


def find_the_top_k_v2(tuple_gnr, top_k, get_max=True):
    """Built-in heapq-based top-k using tuple (key, obj)."""
    if get_max:
        return heapq.nlargest(top_k, tuple_gnr, key=lambda x: x[0])
    else:
        return heapq.nsmallest(top_k, tuple_gnr, key=lambda x: x[0])


def load_ndjson_and_find_the_top_k_v2(input_ndjson_path, top_k, value_type, get_max, use_filter):
    """Top-k using (score, dict) tuple and heapq best practices."""
    dict_gnr = get_dict_gnr_read_ndjson_file_multi_lines(
        input_ndjson_path=input_ndjson_path,
        use_filter=use_filter,
    )
    comparable_tuple_gnr = get_gnr_comparable_tuple(
        dict_gnr=dict_gnr,
        value_type=value_type,
    )
    top_k_tuple = find_the_top_k_v2(
        tuple_gnr=comparable_tuple_gnr,
        top_k=top_k,
        get_max=get_max,
    )
    return list_of_comparable_tuple_to_list_of_original_data(top_k_tuple)


def get_comparable_tuple(dic, value_type):
    """Create (score, dict) tuple for heap comparison."""
    key = list(dic)[0]
    if value_type == "scalar":
        return dic[key], dic
    else:
        return dic[key][0], dic


def get_gnr_comparable_tuple(dict_gnr, value_type):
    """Yield (score, dict) tuples from input generator."""
    for item in dict_gnr:
        yield get_comparable_tuple(item, value_type)


def list_of_comparable_tuple_to_list_of_original_data(lst):
    """Extract original data from (score, data) tuples."""
    _, rst = zip(*lst)
    return list(rst)


def get_dict_gnr_read_ndjson_file_multi_lines(input_ndjson_path, use_filter=False):
    """Yield dict from each line of the NDJSON file."""
    with open(input_ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            yield parse_one_line(
                line,
                use_filter=use_filter,
            )


def find_top_k_and_print(input_ndjson_path, top_k, value_type, get_max, use_filter):
    """Load NDJSON and print top-k results based on file type."""
    rst = load_ndjson_and_find_the_top_k_v2(
        input_ndjson_path=input_ndjson_path,
        top_k=top_k,
        value_type=value_type,
        get_max=get_max,
        use_filter=use_filter,
    )

    if get_max:
        print_info = f"Happiest {top_k} "
    else:
        print_info = f"Saddest {top_k} "

    if "hour_score" in os.path.basename(input_ndjson_path):
        print_info += "hours:"
    elif "id_score" in os.path.basename(input_ndjson_path):
        print_info += "users:"
    else:
        raise NotImplementedError

    print(print_info)
    pprint.pprint(rst)
    print()


def get_value_type(file_name):
    """Infer value type (scalar/list) from filename."""
    if "hour_score" in file_name:
        return "scalar"
    elif "id_score" in file_name:
        return "list"
    else:
        raise NotImplementedError("Cannot infer value type from file name.")


def high_level_api_sort_result():
    """High-level sort runner for top-k on hour/id scores."""
    merged_hour_score_path = next(filter_file(TEST_DATA_FOLDER.glob("merged_hour_score_v?.ndjson")))
    merged_id_score_path = next(filter_file(TEST_DATA_FOLDER.glob("merged_id_score_v?.ndjson")))

    for path in [merged_hour_score_path, merged_id_score_path]:
        basename = path.stem
        value_type = get_value_type(basename)
        for get_max in [True, False]:
            find_top_k_and_print(
                path,
                top_k=5,
                value_type=value_type,
                get_max=get_max,
                use_filter=False,
            )


def filter_file(lst):
    """Filter out non-file items from a Path list."""
    return filter(lambda x: x.is_file(), lst)
