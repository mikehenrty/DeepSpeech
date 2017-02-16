from os import path
from tensorflow.contrib.learn.python.learn.datasets import base
from util.importers.datasets import DataSet, DataSets


def read_data_sets(data_dir, train_batch_size, dev_batch_size, test_batch_size, numcep, numcontext, thread_count=1, limit_dev=0, limit_test=0, limit_train=0, sets=[]):
    # Conditionally download data
    LDC93S1_BASE = "LDC93S1"
    LDC93S1_BASE_URL = "https://catalog.ldc.upenn.edu/desc/addenda/"
    local_file = base.maybe_download(LDC93S1_BASE + ".wav", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".wav")
    _ = base.maybe_download(LDC93S1_BASE + ".txt", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt")
    
    wav_file = path.join(data_dir, "LDC93S1.wav")
    wav_filesize = path.getsize(wav_file)
    txt_file = path.join(data_dir, "LDC93S1.txt")
    with open(txt_file, "r") as open_txt_file:
        transcript = ' '.join(open_txt_file.read().strip().lower().split(' ')[2:]).replace('.', '')

    filelist = pandas.DataFrame(data=[(wav_file, wav_filesize, transcript)])

    # Create all DataSets, we do not really need separation
    train = None
    if "train" in sets:
        train = DataSet(filelist, thread_count, batch_size, numcep, numcontext)
    
    dev = None
    if "dev" in sets:
        dev   = DataSet(filelist, thread_count, batch_size, numcep, numcontext)
    
    test = None
    if "test" in sets:
        test  = DataSet(filelist, thread_count, batch_size, numcep, numcontext)

    # Return DataSets
    return DataSets(train, dev, test)
