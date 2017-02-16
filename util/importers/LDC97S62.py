import fnmatch
import os
import subprocess
import wave
import tensorflow as tf
import unicodedata
import codecs
import pandas

from glob import glob
from util.importers.datasets import DataSet, DataSets

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
