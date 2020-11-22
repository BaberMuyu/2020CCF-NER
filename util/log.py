import time
# import visdom
import numpy as np
import datetime

from global_config import ROOT_LOG


class ModelAux(object):
    def __init__(self, config, train_steps):
        self.config = config
        self.train_steps = train_steps

        self.log_printer = LogPrinter(self.config, self.config.fitting.epochs, self.train_steps)
        # self.vis = visdom.Visdom(env='EL', port=40406) if False else None
        self.vis = None
        self.new_line_flag = False

    def new_line(self):
        self.new_line_flag = True

    def show_log(self, epoch, step, logs):
        global_step = epoch * self.train_steps + step
        if self.vis is not None:
            self.vis.line(X=np.reshape(np.array(global_step), [1]),
                          Y=np.reshape(logs['loss'], [1]),
                          win='loss', name='train_loss',
                          update='append' if global_step else None)
            self.vis.line(X=np.reshape(np.array(global_step), [1]),
                          Y=np.reshape(logs['dev_loss'], [1]),
                          win='loss', name='dev_loss',
                          update='append' if global_step else None)
            self.vis.line(X=np.reshape(np.array(global_step), [1]),
                          Y=np.reshape(logs['f1'], [1]),
                          win='f1', name='train_f1',
                          update='append' if global_step else None)
            self.vis.line(X=np.reshape(np.array(global_step), [1]),
                          Y=np.reshape(logs['dev_f1'], [1]),
                          win='f1', name='dev_f1',
                          update='append' if global_step else None)

        # print log and clean logs
        self.log_printer(epoch, step, logs, self.new_line_flag)
        self.new_line_flag = False


class LogPrinter(object):
    def __init__(self, config, epochs, steps):
        self.epochs = epochs
        self.steps = steps
        self.begin_time = 0
        self.time_list_size = 250
        self.time_list = []
        now = datetime.datetime.now()
        print(now.year)
        print(now.month)
        print(now.day)
        print(now.hour)
        print(now.minute)
        self.file = ROOT_LOG + 'log_{:02}_{:02}_{:02}_{:02}_{:02}.txt'.format(now.month, now.day, now.hour, now.minute, now.second)
        self.fp = open(self.file, 'w+', encoding='utf-8')
        config_str = str(config)
        self.fp.write(config_str)

    @staticmethod
    def get_time_str(second):
        hour_unit = 60 * 60
        min_unit = 60
        h = int(second / hour_unit)
        second = second % hour_unit
        m = int(second / min_unit)
        s = second % min_unit

        time_string = ""
        if h > 0:
            time_string += "%d:%02d:%02d" % (h, m, s)
        elif m > 0:
            time_string += "%d:%02d" % (m, s)
        else:
            time_string += "%ds" % s
        return time_string

    def __call__(self, epoch, step, logs, new_line):
        epoch += 1
        step += 1
        log_str = ""
        log_keys = logs.keys()

        current_time = time.time()
        if step == 1:
            self.begin_time = current_time
            self.time_list = [current_time] * self.time_list_size
        self.time_list = self.time_list[1:]
        self.time_list.append(current_time)

        if step == 1:
            log_str += "\n\nEpoch: %4d/%-6d" % (epoch, self.epochs)
            log_str += '=' * (80 - len(log_str)) + '\n'

        if step == 1:
            _time = -1
        elif step == self.steps:
            # ALL
            _time = current_time - self.begin_time
        elif step < self.time_list_size:
            _time = (self.time_list[-1] - self.time_list[0]) * (self.steps - step) / (step - 1)
        else:
            # ETA
            _time = (self.time_list[-1] - self.time_list[0]) * (self.steps - step) / self.time_list_size

        log_str += '%d/%d' % (step, self.steps)
        log_str += ' - ETA: ' if step != self.steps else ' - ALL: '
        log_str += self.get_time_str(_time) if _time != -1 else "xx:xx:xx"

        # loss and metrics
        for key in log_keys:
            log_str += " - %s: " % key
            num = "{:.6f}".format(logs[key])
            log_str += num[:8]

        self.fp.write(log_str + '\n')
        log_str = '\r' + log_str
        if new_line:
            log_str += '\n'

        print(log_str, end='')


class MovingData(object):
    def __init__(self, window):
        self.window = window
        self.data_dicts = {}
        self.moving_data = {}

    def __call__(self, globle_step, new_data):
        moving_index = int(globle_step % self.window)
        for key in new_data.keys():
            if key not in self.moving_data.keys():
                self.data_dicts[key] = [0] * self.window
                self.moving_data[key] = 0
            self.moving_data[key] += new_data[key] - self.data_dicts[key][moving_index]
            self.data_dicts[key][moving_index] = new_data[key]
        return self.moving_data

# def paint_curve(log_file):
#     fp = open(log_file, 'r', encoding='utf-8')
#     loss = []
#     label_acc = []
#     lines = fp.readlines()
#     fp.close()
#     for line in lines:
#         data = line.split(' - ')
#         if len(data) > 2:
#             loss.append(float(data[2].split(': ')[1]))
#             label_acc.append(float(data[3].split(': ')[1]))
#     step = [i for i in range(len(loss))]
#     plt.plot(step, loss)
#     # plt.show()
#     plt.savefig('scatter.eps',dpi=600,format='eps')
#
# if __name__ == '__main__':
#     paint_curve(ROOT_RESULT + 'temp.txt')
