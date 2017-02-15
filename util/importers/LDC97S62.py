import fnmatch
import os
import subprocess
import wave
import tensorflow as tf
import unicodedata
import codecs
import pandas

from glob import glob
from itertools import cycle
from math import ceil
from Queue import PriorityQueue
from threading import Thread
from util.audio import audiofile_to_input_vector
from util.gpu import get_available_gpus
from util.text import text_to_char_array, ctc_label_dense_to_sparse, validate_label

class DataSets(object):
    def __init__(self, train, dev, test):
        self._dev = dev
        self._test = test
        self._train = train

    def start_queue_threads(self, session):
        self._dev.start_queue_threads(session)
        self._test.start_queue_threads(session)
        self._train.start_queue_threads(session)

    @property
    def train(self):
        return self._train

    @property
    def dev(self):
        return self._dev

    @property
    def test(self):
        return self._test


class DataSet(object):
    def __init__(self, filelist, thread_count, batch_size, numcep, numcontext):
        self._coord = None
        self._numcep = numcep
        self._x = tf.placeholder(tf.float32, [None, numcep + (2 * numcep * numcontext)])
        self._x_length = tf.placeholder(tf.int32, [])
        self._y = tf.placeholder(tf.int32, [None, ])
        self._y_length = tf.placeholder(tf.int32, [])
        self._example_queue = tf.PaddingFIFOQueue(shapes=[[None, numcep + (2 * numcep * numcontext)], [], [None, ], []],
                                                  dtypes = [tf.float32, tf.int32, tf.int32, tf.int32],
                                                  capacity = 2 * self._get_device_count() * batch_size)
        self._enqueue_op = self._example_queue.enqueue([self._x, self._x_length, self._y, self._y_length])
        self._close_op = self._example_queue.close(cancel_pending_enqueues=True)
        self._filelist = filelist
        self._batch_size = batch_size
        self._numcontext = numcontext
        self._thread_count = thread_count
        self._files_circular_list = self._create_files_circular_list()

    def _get_device_count(self):
        available_gpus = get_available_gpus()
        return  max(len(available_gpus), 1)

    def start_queue_threads(self, session, coord):
        self._coord = coord
        batch_threads = [Thread(target=self._populate_batch_queue, args=(session,)) for i in xrange(self._thread_count)]
        for batch_thread in batch_threads:
            batch_thread.daemon = True
            batch_thread.start()
        return batch_threads

    def close_queue(self, session):
        session.run(self._close_op)

    def _create_files_circular_list(self):
        # 1. Sort by wav filesize
        # 2. Select just wav filename and transcript columns
        # 3. Create a cycle
        return cycle(self._filelist.sort_values(by="wav_filesize")
                                   .ix[:, ["wav_filename", "transcript"]]
                                   .itertuples(index=False))

    def _populate_batch_queue(self, session):
        for wav_file, transcript in self._files_circular_list:
            if self._coord.should_stop():
                return
            source = audiofile_to_input_vector(wav_file, self._numcep, self._numcontext)
            source_len = len(source)
            target = text_to_char_array(transcript)
            target_len = len(target)
            try:
                session.run(self._enqueue_op, feed_dict={
                    self._x: source,
                    self._x_length: source_len,
                    self._y: target,
                    self._y_length: target_len})
            except tf.errors.CancelledError:
                return

    def next_batch(self):
        source, source_lengths, target, target_lengths = self._example_queue.dequeue_many(self._batch_size)
        sparse_labels = ctc_label_dense_to_sparse(target, target_lengths, self._batch_size)
        return source, source_lengths, sparse_labels

    @property
    def total_batches(self):
        # Note: If len(_filelist) % _batch_size != 0, this re-uses initial files
        return int(ceil(float(len(self._filelist)) /float(self._batch_size)))


def read_data_sets(data_dir, train_batch_size, dev_batch_size, test_batch_size, numcep, numcontext, thread_count=8, limit_dev=0, limit_test=0, limit_train=0, sets=[]):
    data_dir = os.path.join(data_dir, "LDC97S62")

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

        # Conditionally convert swb sph data to wav
        _maybe_convert_wav(data_dir, "swb1_d1", "swb1_d1-wav")
        _maybe_convert_wav(data_dir, "swb1_d2", "swb1_d2-wav")
        _maybe_convert_wav(data_dir, "swb1_d3", "swb1_d3-wav")
        _maybe_convert_wav(data_dir, "swb1_d4", "swb1_d4-wav")

        # Conditionally split wav data
        filelist = _maybe_split_wav_and_trans(data_dir,
                         "swb_ms98_transcriptions",
                         "swb1_d1-wav",
                         "swb1_d1-split-wav")
        filelist.append(_maybe_split_wav_and_trans(data_dir,
                            "swb_ms98_transcriptions",
                            "swb1_d2-wav",
                            "swb1_d2-split-wav"))
        filelist.append(_maybe_split_wav_and_trans(data_dir,
                            "swb_ms98_transcriptions",
                            "swb1_d3-wav",
                            "swb1_d3-split-wav"))
        filelist.append(_maybe_split_wav_and_trans(data_dir,
                            "swb_ms98_transcriptions",
                            "swb1_d4-wav",
                            "swb1_d4-split-wav"))

        train_files, dev_files, test_files = _split_sets(filelist)

        # Write processed sets to disk as CSV files
        train_files.to_csv(train_csv, index=False)
        dev_files.to_csv(dev_csv, index=False)
        test_files.to_csv(test_csv, index=False)

    # Create dev DataSet
    dev = None
    if "dev" in sets:
        dev = _read_data_set(dev_files, thread_count, dev_batch_size, numcep,
                             numcontext, limit_dev)

    # Create test DataSet
    test = None
    if "test" in sets:
        test = _read_data_set(test_files, thread_count, test_batch_size, numcep,
                             numcontext, limit_test)

    # Create train DataSet
    train = None
    if "train" in sets:
        train = _read_data_set(train_files, thread_count, train_batch_size, numcep,
                             numcontext, limit_train)

    # Return DataSets
    return DataSets(train, dev, test)

def _maybe_convert_wav(data_dir, original_data, converted_data):
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)

    # Conditionally convert sph files to wav files
    if os.path.exists(target_dir):
        print("skipping maybe_convert_wav")
        return

    # Create target_dir
    os.makedirs(target_dir)

    # Loop over sph files in source_dir and convert each to 16-bit PCM wav
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.sph"):
            for channel in ['1', '2']:
                sph_file = os.path.join(root, filename)
                wav_filename = os.path.splitext(os.path.basename(sph_file))[0] + "-" + channel + ".wav"
                wav_file = os.path.join(target_dir, wav_filename)
                print("converting {} to {}".format(sph_file, wav_file))
                subprocess.check_call(["sph2pipe", "-c", channel, "-p", "-f", "rif", sph_file, wav_file])

def _parse_transcriptions(trans_file):
    segments = []
    with codecs.open(trans_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.startswith("#")  or len(line) <= 1:
                continue

            filename_time_beg = 0;
            filename_time_end = line.find(" ", filename_time_beg)

            start_time_beg = filename_time_end + 1
            start_time_end = line.find(" ", start_time_beg)

            stop_time_beg = start_time_end + 1
            stop_time_end = line.find(" ", stop_time_beg)

            transcript_beg = stop_time_end + 1
            transcript_end = len(line)

            transcript = line[transcript_beg:transcript_end].strip().lower()
            transcript = unicodedata.normalize("NFKD", transcript)
                                    .encode("ascii", "ignore")

            if validate_label(transcript) == None:
                continue

            segments.append({
                "start_time": float(line[start_time_beg:start_time_end]),
                "stop_time": float(line[stop_time_beg:stop_time_end]),
                "speaker": line[6],
                "transcript": transcript,
            })
    return segments


def _maybe_split_wav_and_trans(data_dir, trans_data, original_data, converted_data):
    trans_dir = os.path.join(data_dir, trans_data)
    source_dir = os.path.join(data_dir, original_data)
    target_dir = os.path.join(data_dir, converted_data)
    if os.path.exists(target_dir):
        print("skipping maybe_split_wav")
        return

    os.makedirs(target_dir)
    
    files = []

    # Loop over transcription files and split corresponding wav
    for root, dirnames, filenames in os.walk(trans_dir):
        for filename in fnmatch.filter(filenames, "*.text"):
            if "trans" not in filename:
                continue
            trans_file = os.path.join(root, filename)
            segments = _parse_transcriptions(trans_file)

            # Open wav corresponding to transcription file
            channel = ("2","1")[(os.path.splitext(os.path.basename(trans_file))[0])[6] == 'A']
            wav_filename = "sw0" + (os.path.splitext(os.path.basename(trans_file))[0])[2:6] + "-" + channel + ".wav"
            wav_file = os.path.join(source_dir, wav_filename)

            print("splitting {} according to {}".format(wav_file, trans_file))

            if not os.path.exists(wav_file):
                print("skipping. does not exist:" + wav_file)
                continue

            origAudio = wave.open(wav_file, "r")

            # Loop over segments and split wav_file for each segment
            for segment in segments:
                # Create wav segment filename
                start_time = segment["start_time"]
                stop_time = segment["stop_time"]
                new_wav_filename = os.path.splitext(os.path.basename(trans_file))[0] + "-" + str(
                    start_time) + "-" + str(stop_time) + ".wav"
                new_wav_file = os.path.join(target_dir, new_wav_filename)
                _split_wav(origAudio, start_time, stop_time, new_wav_file)

                new_wav_filesize = os.path.getsize(new_wav_file)
                transcript = segment["transcript"]

                files.append((new_wav_file, new_wav_filesize, transcript))

            # Close origAudio
            origAudio.close()

    return pandas.DataFrame(data=files,
                        columns=["wav_filename", "wav_filesize", "transcript"])

def _split_wav(origAudio, start_time, stop_time, new_wav_file):
    frameRate = origAudio.getframerate()
    origAudio.setpos(int(start_time * frameRate))
    chunkData = origAudio.readframes(int((stop_time - start_time) * frameRate))
    chunkAudio = wave.open(new_wav_file, "w")
    chunkAudio.setnchannels(origAudio.getnchannels())
    chunkAudio.setsampwidth(origAudio.getsampwidth())
    chunkAudio.setframerate(frameRate)
    chunkAudio.writeframes(chunkData)
    chunkAudio.close()

def _split_sets(filelist):
    # We initially split the entire set into 80% train and 20% test, then
    # split the train set into 80% train and 20% validation.
    train_beg = 0
    train_end = int(0.8 * len(filelist))

    dev_beg = int(0.8 * train_end)
    dev_end = train_end
    train_end = dev_beg

    test_beg = dev_end
    test_end = len(filelist)

    return filelist.iloc[train_beg:train_end],
           filelist.iloc[dev_beg:dev_end],
           filelist.iloc[test_beg:test_end]

def _read_data_set(filelist, thread_count, batch_size, numcep, numcontext, limit=0):
    # Optionally apply dataset size limit
    if limit > 0:
        filelist = filelist.iloc[:limit]

    # Return DataSet
    return DataSet(filelist, thread_count, batch_size, numcep, numcontext)
