import subprocess

try:
    import telemetry
    ngc_telemetry = telemetry.ApplicationTelemetry()
except:
    ngc_telemetry = False
    print("Could not load NGC telemetry!")


def push_ngc_telemetry(name, value):
    # if NGC telemetry logging enabled:
    try:
        ngc_telemetry.metric_push_async({'metric': name, 'value': value})
    except:
        pass


def log_ngc(train_metric):
    for key in train_metric.keys():
        push_ngc_telemetry(key, train_metric[key])


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
                'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [x for x in result.strip().split('\n')]
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def get_gpu_utilization():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
                'nvidia-smi', '--query-gpu=utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def log_gpu_statistics():
    memory = get_gpu_memory_map()
    utilization = get_gpu_utilization()

    printstring="\n"

    ngpu=1
    for mem, use in zip(memory, utilization):
        printstring += "GPU{}: memory {} ({}%); ".format(ngpu, mem, use)

        try:
            log_ngc("GPU{} memory".format(ngpu), mem)
            log_ngc("GPU{} usage".format(ngpu), use)
        except:
            pass

    print(printstring)

def log_ngc_dict(metric, prefix):
    for key in metric.keys():
        push_ngc_telemetry("{prefix}-{key}".format(prefix=prefix, key=key), metric[key])