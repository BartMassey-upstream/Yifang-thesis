import os
import random
import re
import numpy as np
import sys
from pathlib import Path
from python_speech_features import mfcc
from tslearn.metrics import dtw
from scipy.io import wavfile
from collections import Counter
import pickle
import heapq

TIMIT = Path("/Users/zhuyifang/Downloads/archive")
#TIMIT = Path("/home/bart/work/reed-theses/zhu-thesis/timit")


class Phone():

    def __init__(self, samplerate, data, transcription) -> None:
        self.samplerate = samplerate
        self.data = data
        self.mfcc_seq = None
        self.transcription = transcription

    def __str__(self) -> str:
        return f"samplerate = {self.samplerate}, data = {self.data}, \
            transcription = {self.transcription}, mfcc_seq = {self.mfcc_seq}"

    def get_mfcc_seq(self):
        self.mfcc_seq = mfcc(self.data, self.samplerate)
        return self.mfcc_seq

    def distance_to(self, other: "Phone"):
        return dtw(self.mfcc_seq, other.mfcc_seq)


class File():

    def __init__(self, path, name) -> None:
        self.path = path
        self.name = name  # filename without extension
        self.wav = np.array([])
        self.samplerate = 16000
        self.phn = []

    def __str__(self) -> str:
        return f"path = {self.path}, name = {self.name}, wav = {self.wav}, \
            phn = {self.phn}"

    def get_phones(self):
        phones = []
        for line in self.phn:
            start, end, transcription = line.split()
            start, end = int(start), int(end)
            wav_data = self.wav[start:end]
            phone = Phone(self.samplerate, wav_data, transcription)
            phones.append(phone)
        return phones


# root should be given as the absolute path
# return files
def get_all_matched_files(root: str) -> list[File]:
    phn_re = re.compile(r".+\.PHN")
    matched_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if phn_re.match(filename):
                filename = filename[:-4]
                file = File(dirpath, filename)
                matched_files.append(file)
    return matched_files


# read .wav and .PHN
def read_files(files: list[File]):
    for file in files:
        filepath = os.path.join(file.path, file.name)
        # read .PHN
        with open(filepath + ".PHN") as f:
            file.phn = f.readlines()
        # read .wav
        file.samplerate, file.wav = wavfile.read(filepath + ".WAV.wav")
    return files


# read all the files in the training set and make them into Phone objects
def get_phones(TIMIT_path, set_name):
    set_path = TIMIT_path / f"data/{set_name}"
    set_files = get_all_matched_files(set_path)
    read_files(set_files)
    set_phones = []
    for file in set_files:
        set_phones += file.get_phones()
    for phone in set_phones:
        phone.get_mfcc_seq()
    print(f"set parse finished: {len(set_phones)} phones")
    return set_phones


# save the phones into a file
def save_phones(phones, filename):
    with open(filename, "wb") as f:
        pickle.dump(phones, f)


# read phones from a file
def read_phones(filename):
    with open(filename, "rb") as f:
        phones = pickle.load(f)
    return phones


def test_accuracy(train_set_phones, test_set_phones):
    # test accuracy
    correct_num = 0

    # iterate all the phones in the test set
    # nphones = int(sys.argv[1])
    nphones = 100
    test_phones = random.sample(test_set_phones, nphones)
    for test_phone in test_phones:
        test_phone.get_mfcc_seq()
        # using KNN to find the nearest neighbor
        k = 10
        # using a heap to keep track of the samllest k element
        # the items in the heap are tuples like (negative distance to the test_set_phone, train_set_phone transcription)
        heap = []
        heapq.heapify(heap)
        for train_set_phone in train_set_phones:
            distance = test_phone.distance_to(train_set_phone)
            if len(heap) < k:
                heapq.heappush(heap,
                               (-distance, train_set_phone.transcription))
            else:
                if -heap[0][0] > distance:
                    heapq.heapreplace(
                        heap, (-distance, train_set_phone.transcription))

        # using Counter to get the most common phone in the heap
        counter = Counter()
        for i in range(k):
            _, transcription = heapq.heappop(heap)
            counter[transcription] += 1

        # predicted_phone is the most common phone in the heap
        predicted_phone = counter.most_common(1)[0][0]

        # if the prediction is correct
        if predicted_phone == test_phone.transcription:
            correct_num += 1
            print("correct")
        # if the prediction is wrong
        else:
            print(
                f'predicted phone is: {predicted_phone}, actual phone is: {test_phone.transcription}'
            )
    # print the accuracy
    print(f"Accuracy: {correct_num / nphones}")


if __name__ == "__main__":

    # if test_set_phones.pkl and train_set_phones.pkl are not created
    # run the following code to create them
    if not Path("test_set_phones.pkl").exists() or not Path(
            "train_set_phones.pkl").exists():

        # read all the files in the training set and make them into Phone objects
        train_set_phones = get_phones(TIMIT, "TRAIN")

        # save the train_set_phones to a file
        save_phones(train_set_phones, "train_set_phones.pkl")

        # read all the files in the testing set and make them into Phone objects
        test_set_phones = get_phones(TIMIT, "TEST")

        # save the test_set_phones to a file
        save_phones(test_set_phones, "test_set_phones.pkl")

    else:
        # read the train_set_phones from a file
        train_set_phones = read_phones("train_set_phones.pkl")

        # read the test_set_phones from a file
        test_set_phones = read_phones("test_set_phones.pkl")

    # test accuracy
    test_accuracy(train_set_phones, test_set_phones)