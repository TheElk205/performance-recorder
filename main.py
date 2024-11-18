import argparse
import signal
import sys
import textwrap
from time import sleep, time
import pickle

import matplotlib.pyplot as plt

import nvidia_smi
import psutil

cpu = []
memory = []
swap = []
gpu_mem = []
gpu_util = []
timestamps = []

def handler(signum, frame):
    global args, cpu, memory, swap, gpu_mem, gpu_util, timestamps
    print("Aborting, saving to file. Stripping all values to the same length")
    use_gpu = args.record_nvda_gpu
    min_length = min(len(cpu), len(memory), len(swap), len(timestamps))

    if use_gpu:
        min_length = min(min_length, len(gpu_mem), len(gpu_util))
        gpu_util = gpu_util[:min_length]
        gpu_mem = gpu_mem[:min_length]

    cpu = cpu[:min_length]
    memory = memory[:min_length]
    swap = swap[:min_length]
    timestamps = timestamps[:min_length]

    save_to_file_and_plot(use_gpu)
    sys.exit(0)

def save_to_file_and_plot(record_nvda_gpu):
    global cpu, memory, swap, gpu_mem, gpu_util, timestamps
    data = {
        "timestamps": timestamps,
        "cpu": cpu,
        "memory": memory,
        "swap": swap,
        "gpu_mem": gpu_mem,
        "gpu_util": gpu_util
    }

    print("Writing to file")
    with open('{}_data.pkl'.format(time()), 'wb') as f:
        pickle.dump(data, f)

    print("Plotting")
    if record_nvda_gpu:
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_total = sizeof_fmt(mem.total)

    memory_total = sizeof_fmt(psutil.virtual_memory().total)
    swap_memory_total = sizeof_fmt(psutil.swap_memory().total)

    plt.title("Performance")
    plt.plot(data["timestamps"], data["memory"], label="Memory ({})".format(memory_total))
    plt.plot(data["timestamps"], data["cpu"], label="CPU")
    plt.plot(data["timestamps"], data["swap"], label="Swap ({})".format(swap_memory_total))
    if record_nvda_gpu:
        plt.plot(data["timestamps"], data["gpu_mem"], label="GPU Mem ({})".format(gpu_mem_total))
        plt.plot(data["timestamps"], data["gpu_util"], label="GPU Usage")
    plt.legend()
    plt.show()

def record_ongoing(seconds = 10, interval_seconds = 1.0, record_gpu= False):
    global cpu, memory, swap, gpu_mem, gpu_util, timestamps

    start = time()
    while (seconds <=0) or (time() - start < seconds):
        cycle_start = time()
        timestamps.append(cycle_start)
        cpu.append(psutil.cpu_percent(interval=interval_seconds/2, percpu=False))
        memory.append(psutil.virtual_memory().percent)
        swap.append(psutil.swap_memory().percent)

        # GPU
        if record_gpu:
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)

            gpu_mem.append(util.memory)
            gpu_util.append(util.gpu)

        duration = time()-cycle_start
        # print(duration)
        sleep(max(0.0, interval_seconds - duration))

def sizeof_fmt(num, suffix="B"):
    # from: https://stackoverflow.com/a/1094933
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

signal.signal(signal.SIGINT, handler)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="System stats visualizer",
        epilog=textwrap.dedent('''\
        Visualizes used system resources after a given amount of time has elapsed. \
        Mainly used for remote evaluation during model training.
        '''
                               ))
    parser.add_argument('--recording-time', type=int, default=0, help="Duration in seconds we want to record, If <=0, runs until ctrl+c is pressed." )
    parser.add_argument('--interval-seconds', type=float, default=1.0, help="Reading interval in seconds we want to achieve")
    parser.add_argument('--record-nvda-gpu', type=bool, default=False, help="If present, NVIDIA GPU Memory and usage will be collected as well")

    args = parser.parse_args()

    if args.recording_time <= 0:
        print("Running indefinitely")
    else:
        print("Running for at most {} seconds.".format(args.recording_time))
    print("Evaluating NVIDIA GPU: {}".format(args.record_nvda_gpu))
    record_ongoing(args.recording_time, args.interval_seconds)

    save_to_file_and_plot(args.record_nvda_gpu)

    plt.plot([1, 2, 4, 6, 8, 5, 4])
    plt.show()