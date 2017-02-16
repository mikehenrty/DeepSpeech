import codecs
import pandas
import tarfile
import tensorflow as tf
import unicodedata
import wave

from os import makedirs
from os import path
from os import remove
from os import rmdir
from sox import Transformer
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
from util.importers.datasets import DataSet, DataSets
from util.stm import parse_stm_file


def read_data_sets(data_dir, train_batch_size, dev_batch_size, test_batch_size, numcep, numcontext, thread_count=8, limit_dev=0, limit_test=0, limit_train=0, sets=[]):
    # Read the processed set files from disk if they exist, otherwise create
    # them.
    train_files = None
    train_csv = os.path.join(data_dir, "ted-train.csv")
    if gfile.Exists(train_csv):
        train_files = pandas.read_csv(train_csv)

    dev_files = None
    dev_csv = os.path.join(data_dir, "ted-dev.csv")
    if gfile.Exists(dev_csv):
        dev_files = pandas.read_csv(dev_csv)

    test_files = None
    test_csv = os.path.join(data_dir, "ted-test.csv")
    if gfile.Exists(test_csv):
        test_files = pandas.read_csv(test_csv)

    if train_files is None or dev_files is None or test_files is None:
        print("Processed dataset files not found, downloading and processing data...")

        # Conditionally download data
        TED_DATA = "TEDLIUM_release2.tar.gz"
        TED_DATA_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"
        local_file = base.maybe_download(TED_DATA, data_dir, TED_DATA_URL)

        # Conditionally extract TED data
        TED_DIR = "TEDLIUM_release2"
        _maybe_extract(data_dir, TED_DIR, local_file)

        # Conditionally convert TED sph data to wav
        _maybe_convert_wav(data_dir, TED_DIR)

        # Conditionally split TED wav and stm data
        train_files, dev_files, test_files = _maybe_split_wav_and_stm(data_dir, TED_DIR)

        # Write processed sets to disk as CSV files
        train_files.to_csv(train_csv, index=False)
        dev_files.to_csv(dev_csv, index=False)
        test_files.to_csv(test_csv, index=False)

    # Create dev DataSet
    dev = None
    if "dev" in sets:
        dev = _read_data_set(dev_files, thread_count, dev_batch_size, numcep, numcontext, limit=limit_dev)

    # Create test DataSet
    test = None
    if "test" in sets:
        test = _read_data_set(test_files, thread_count, test_batch_size, numcep, numcontext, limit=limit_test)

    # Create train DataSet
    train = None
    if "train" in sets:
        train = _read_data_set(train_files, thread_count, train_batch_size, numcep, numcontext, limit=limit_train)

    # Return DataSets
    return DataSets(train, dev, test)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(path.join(data_dir, extracted_data)):
      tar = tarfile.open(archive)
      tar.extractall(data_dir)
      tar.close()

def _maybe_convert_wav(data_dir, extracted_data):
    # Create extracted_data dir
    extracted_dir = path.join(data_dir, extracted_data)

    # Conditionally convert dev sph to wav
    _maybe_convert_wav_dataset(extracted_dir, "dev")

    # Conditionally convert train sph to wav
    _maybe_convert_wav_dataset(extracted_dir, "train")

    # Conditionally convert test sph to wav
    _maybe_convert_wav_dataset(extracted_dir, "test")

def _maybe_convert_wav_dataset(extracted_dir, data_set):
    # Create source dir
    source_dir = path.join(extracted_dir, data_set, "sph")

    # Create target dir
    target_dir = path.join(extracted_dir, data_set, "wav")

    # Conditionally convert sph files to wav files
    if not gfile.Exists(target_dir):
        # Create target_dir
        makedirs(target_dir)

        # Loop over sph files in source_dir and convert each to wav
        for sph_file in glob(path.join(source_dir, "*.sph")):
            transformer = Transformer()
            wav_filename = path.splitext(path.basename(sph_file))[0] + ".wav"
            wav_file = path.join(target_dir, wav_filename)
            transformer.build(sph_file, wav_file)
            remove(sph_file)

        # Remove source_dir
        rmdir(source_dir)

def _maybe_split_wav_and_stm(data_dir, extracted_data):
    # Create extracted_data dir
    extracted_dir = path.join(data_dir, extracted_data)

    # Conditionally split dev wav
    dev_files = _maybe_split_dataset(extracted_dir, "dev")

    # Conditionally split train wav
    train_files = _maybe_split_dataset(extracted_dir, "train")

    # Conditionally split test wav
    test_files = _maybe_split_dataset(extracted_dir, "test")

    return train_files, dev_files, test_files

def _maybe_split_dataset(extracted_dir, data_set):
    # Create stm dir
    stm_dir = path.join(extracted_dir, data_set, "stm")

    # Create wav dir
    wav_dir = path.join(extracted_dir, data_set, "wav")

    files = []

    # Loop over stm files and split corresponding wav
    for stm_file in glob(path.join(stm_dir, "*.stm")):
        # Parse stm file
        stm_segments = parse_stm_file(stm_file)

        # Open wav corresponding to stm_file
        wav_filename = path.splitext(path.basename(stm_file))[0] + ".wav"
        wav_file = path.join(wav_dir, wav_filename)
        origAudio = wave.open(wav_file,'r')

        # Loop over stm_segments and split wav_file for each segment
        for stm_segment in stm_segments:
            # Create wav segment filename
            start_time = stm_segment.start_time
            stop_time = stm_segment.stop_time
            new_wav_filename = path.splitext(path.basename(stm_file))[0] + "-" + str(start_time) + "-" + str(stop_time) + ".wav"
            new_wav_file = path.join(wav_dir, new_wav_filename)

            # If the wav segment filename does not exist create it
            if not gfile.Exists(new_wav_file):
                _split_wav(origAudio, start_time, stop_time, new_wav_file)

            new_wav_filesize = path.getsize(new_wav_file)
            transcript = stm_segment.transcript
            files.append((new_wav_file, new_wav_filesize, transcript))

        # Close origAudio
        origAudio.close()

        # Remove wav_file
        remove(wav_file)

        # Remove stm_file
        remove(stm_file)

    return pandas.DataFrame(data=files,
                            columns=["wav_filename", "wav_filesize", "transcript"])

def _split_wav(origAudio, start_time, stop_time, new_wav_file):
    frameRate = origAudio.getframerate()
    origAudio.setpos(int(start_time*frameRate))
    chunkData = origAudio.readframes(int((stop_time - start_time)*frameRate))
    chunkAudio = wave.open(new_wav_file,'w')
    chunkAudio.setnchannels(origAudio.getnchannels())
    chunkAudio.setsampwidth(origAudio.getsampwidth())
    chunkAudio.setframerate(frameRate)
    chunkAudio.writeframes(chunkData)
    chunkAudio.close()

def _read_data_set(filelist, thread_count, batch_size, numcep, numcontext, limit=0):
    # Optionally apply dataset size limit
    if limit > 0:
        filelist = filelist.iloc[:limit]

    # Return DataSet
    return DataSet(filelist, thread_count, batch_size, numcep, numcontext)
