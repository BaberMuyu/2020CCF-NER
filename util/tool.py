import time
import json
from multiprocessing import Process, Manager


def count_item(x):
    y = []
    rate = []
    ssum = 0
    for i in x:
        if i > len(y) - 1:
            y = y + [0] * (i - len(y) + 1)
        y[i] += 1
        ssum += 1
    part_sum = 0
    for i, n in enumerate(y):
        part_sum += n
        rate.append((i, n, round(part_sum / ssum, 6), round(n / ssum, 6)))
    return rate


def json_dump(data, file):
    fp = open(file, 'w+', encoding='utf-8')
    json.dump(data, fp, ensure_ascii=False, indent=4)
    fp.flush()
    time.sleep(2)
    fp.close()


def json_dump_by_line(data, file):
    fp = open(file, 'w+', encoding='utf-8')
    for d in data:
        string = json.dumps(d, ensure_ascii=False) + '\n'
        fp.write(string)
    fp.flush()
    time.sleep(2)
    fp.close()


def json_load(file):
    fp = open(file, 'r', encoding='utf-8')
    d = json.load(fp)
    fp.close()
    return d


def json_load_by_line(file):
    data = []
    fp = open(file, 'r', encoding='utf-8')
    for d in fp.readlines():
        d = json.loads(d)
        data.append(d)
    fp.close()
    return data


def simple_multiprocess(func, data, out_data, num_works=1):
    out_list = []
    basic_unit = int(len(data) / num_works)
    p_list = []
    with Manager() as manager:
        for i in range(0, num_works):
            if i == num_works - 1:
                d = data[basic_unit * i:]
            else:
                d = data[basic_unit * i: basic_unit * (i + 1)]
            out_temp = manager.dict([(key, manager.list()) for key in out_data.keys()])
            out_list.append(out_temp)
            p = Process(target=func, args=(d, out_temp))
            p_list.append(p)

        for p in p_list:
            p.start()

        for p in p_list:
            p.join()

        for d in out_list:
            for key in out_data.keys():
                out_data[key].extend(d[key])


def build_tree(data, key):
    tree = {}
    for d in data:
        if d[key] not in tree:
            tree[d[key]] = [d]
        else:
            tree[d[key]].append(d)
    return tree


def calculate_f1(correct_num, pred_num, y_true_num, verbose=False):
    if correct_num == 0 or pred_num == 0 or y_true_num == 0:
        precise = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precise = correct_num / pred_num
        recall = correct_num / y_true_num
        f1 = 2 * precise * recall / (precise + recall)
    if verbose:
        return precise, recall, f1
    else:
        return f1


def kmp(mom_string, son_string):
    # 传入一个母串和一个子串
    # 返回子串匹配上的第一个位置，若没有匹配上返回-1
    if type(mom_string) != type(son_string):
        return -1
    if len(son_string) == 0:
        return -1
    if len(mom_string) == 0:
        return -1
    # 求next数组
    next = [-1] * len(son_string)
    if len(son_string) > 1:  # 这里加if是怕列表越界
        next[1] = 0
        i, j = 1, 0
        while i < len(son_string) - 1:  # 这里一定要-1，不然会像例子中出现next[8]会越界的
            if j == -1 or son_string[i] == son_string[j]:
                i += 1
                j += 1
                next[i] = j
            else:
                j = next[j]

    # kmp框架
    m = s = 0  # 母指针和子指针初始化为0
    while s < len(son_string) and m < len(mom_string):
        # 匹配成功,或者遍历完母串匹配失败退出
        if s == -1 or mom_string[m] == son_string[s]:
            m += 1
            s += 1
        else:
            s = next[s]

    if s == len(son_string):  # 匹配成功
        return m - s
    else:
        return -1


class TimeStatistic(object):
    def __init__(self):
        self.data = {}
        self.last_time = 0
        self.total_time = 0

    def init(self):
        self.data = {}
        self.last_time = time.time()
        self.total_time = 0

    def update(self, s):
        if True:
            current = time.time()
            time_slice = current - self.last_time
            self.last_time = current
            self.total_time += time_slice
            if s not in self.data.keys():
                self.data[s] = time_slice
            else:
                self.data[s] += time_slice

    def show(self):
        for key in self.data.keys():
            print("{}: {}".format(key, self.data[key] / self.total_time))
