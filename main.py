# BCM: Start with a description of this file.
# "Main program for phone recognizer."
# Then continue with your name and the year
# the file was created.
# "Yifang Zhu 2023"
# Do this for every source file.

from pathlib import Path
# BCM: "File" is not a great type name. Something like "PhoneFile"
# would help differentiate it from Python's "file".
from recognizer.file import File
from recognizer.phone import Phone
import re
import os
import pickle
import random
from collections import Counter
import heapq
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import librosa
import csv
import argparse
from collections.abc import Callable

# BCM: Every block should have a comment describing
# what it is and does.
parser = argparse.ArgumentParser(
    prog = 'phonerec',
    description = 'segmented phone recognizer',
)
parser.add_argument(
    '-s',
    '--stretch',
    action = "store_true",
    # BCM: Need to add help fields to all
    # of these indicating what they are:
    # "help = 'timestretch the instances to constant length'"
)
parser.add_argument(
    '-d',
    '--distance',
    default = "dtw",
)
parser.add_argument(
    '--verbose',
    action = "store_true",
)
args = parser.parse_args()

# BCM: Not good practice to hardcode a personal path in the
# codebase. Use the environment variable or add a config
# file or something instead.
TIMIT = Path("/Users/zhuyifang/Downloads/archive")
if "TIMIT" in os.environ:
    TIMIT = Path(os.environ["TIMIT"])

# Explain why these phones are ignored.
IGNORED_PHONES = {"h#", "#h", "sil", "pau", "epi"}

# These names aren't ideal: always be suspicious of
# names that have 1, 2, 3, … in them. Try this:
#     PHONE_GROUPS = [
#         {'axr', 'er'},
#         {'m', 'em'},
#         {'n', 'en', 'nx'},
#         {'ng', 'eng'},
#     ]
GROUP_1 = {'axr', 'er'}
GROUP_2 = {'m', 'em'}
GROUP_3 = {'n', 'en', 'nx'}
GROUP_4 = {'ng', 'eng'}


# BCM: In general, comments in Python should be complete
# sentences starting with a capital and ending with a
# period. See PEP-8 https://peps.python.org/pep-0008/ for a
# very detailed set of guidelines to follow.  I recommend
# the Black code formatter
# https://black.readthedocs.io/en/stable/ for getting code
# into compliance with PEP-8.

# BCM: I don't know what this comment means.
# "root should be given as the absolute path"
# What is the root parameter?
# BCM: This comment is not useful
# "return files"
# The type signature already tells us this.
def get_all_matched_files(root: str) -> list[File]:
    # BCM: Make "phn_re" global so that it only gets
    # compiled once per run.
    # BCM: The suffix itself should also be global,
    # since it may be used in multiple places:
    # "phn_suffix = 'PHN'"
    phn_re = re.compile(r".+\.PHN")
    matched_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            # BCM: ".fullmatch()" is needed here to not match
            # things like "foo.PHNXXX".
            if phn_re.match(filename):
                # BCM: The "-4" here is quite dangerous,
                # since the suffix may change. Use
                # regex submatches as well:
                #     phn_re = re.compile(rf"(.+)\.{phn_suffix}")
                #     matches = phn_re.fullmatch(filename)
                #     if matches:
                #         filename = matches.group(1)
                filename = filename[:-4]
                file = File(dirpath, filename)
                matched_files.append(file)
    return matched_files


# read .wav and .PHN
def read_files(files: list[File]) -> list[File]:
    for file in files:
        # BCM: Suggest using pathlib's Path rather than
        # os.path: it works better and does more stuff.
        # Here, for example you could write "file.path / file.name".
        # But really making a path should be a File method, not here.
        filepath = os.path.join(file.path, file.name)
        # BCM: These comments don't add anything. It is obvious that
        # a PHN is being read here. Also, again use a global for the
        # extension, or make it part of File.
        # "read .PHN"
        with open(filepath + ".PHN") as f:
            file.phn = f.readlines()
        # read .wav
        file.wav, file.samplerate = librosa.load(filepath + ".WAV.wav",
                                                 sr=16000)
    return files


# BCM: Does "TIMIT_path" need to be a parameter? Maybe better to
# use the global? Also, capitalization is a bit rough: "timit_path" is fine.
# read all the files in the training set and make them into Phone objects
def get_phones_from_TIMIT(TIMIT_path: Path, set_name: str) -> list[Phone]:
    set_path = TIMIT_path / f"data/{set_name}"
    set_files = get_all_matched_files(set_path)
    print(f"set parse started: {len(set_files)} files")
    read_files(set_files)
    set_phones = []
    for file in set_files:
        set_phones += file.get_phones()
    for phone in set_phones:
        phone.get_mfcc_seq()
    print(f"set parse finished: {len(set_phones)} phones")
    return set_phones


# save the phones into a file
def save_phones_to_pkl(phones: list[Phone], filename: str):
    with open(filename, "wb") as f:
        pickle.dump(phones, f)

# read phones from a file
def read_phones_from_pkl(filename: str) -> list[Phone]:
    with open(filename, "rb") as f:
        phones = pickle.load(f)
    return phones


# BCM: This function is a miss. That's my fault. This thing
# should be rewritten from scratch to do what it is supposed
# to do.
def get_phones(namer: Callable[[str], str], do_pkl=None) -> tuple[list[Phone], list[Phone], bool]:
    assert do_pkl is not None, "get_phones: pkl required"
    pkls = (
        (Path(namer("train")), "TRAIN"),
        (Path(namer("test")), "TEST"),
    )
    # if test_set_phones.pkl and train_set_phones.pkl are not created
    # run the following code to create them
    tt_phones = []
    pkled = False
    for pkl in pkls:
        pkl_path, timit_dir = pkl
        if not pkl_path.exists():
            # read all the files in the phone set and make them into Phone objects
            phones = get_phones_from_TIMIT(TIMIT, timit_dir)

            # save the phones to a pkl file
            if do_pkl:
                save_phones_to_pkl(phones, pkl_path)

            tt_phones.append(phones)
        else:
            # read the train_set_phones from a file
            phones = read_phones_from_pkl(pkl_path)
            tt_phones.append(phones)
            pkled = True
    return (*tt_phones, pkled)

# BCM: Needs description.
def drop_ignored_phones(phones: list[Phone]) -> list[Phone]:
    return list(
        filter(lambda phone: phone.transcription not in IGNORED_PHONES,
               phones))


# BCM: Needs description.
def group_phones(phones: list[Phone]) -> dict[str, list[Phone]]:
    # BCM: Rather than hand-listing this dictionary, should
    # derive it from the phones and phone_groups.
    res = {
        'ix': [],
        'iy': [],
        's': [],
        'r': [],
        'n/en/nx': [],
        'l': [],
        'tcl': [],
        'kcl': [],
        'ih': [],
        'dcl': [],
        'k': [],
        't': [],
        'm/em': [],
        'eh': [],
        'ae': [],
        'axr/er': [],
        'ax': [],
        'z': [],
        'd': [],
        'q': [],
        'w': [],
        'ao': [],
        'aa': [],
        'dh': [],
        'pcl': [],
        'p': [],
        'dx': [],
        'f': [],
        'b': [],
        'ah': [],
        'ay': [],
        'gcl': [],
        'ey': [],
        'sh': [],
        'ow': [],
        'bcl': [],
        'g': [],
        'v': [],
        'y': [],
        'ux': [],
        'ng/eng': [],
        'jh': [],
        'hv': [],
        'hh': [],
        'el': [],
        'th': [],
        'oy': [],
        'ch': [],
        'uh': [],
        'aw': [],
        'uw': [],
        'ax-h': [],
        'zh': []
    }

    for phone in phones:
        # BCM: Building PHONE_GROUPS earlier pays off here, as this
        # copy-paste can be replaced with a for loop.
        # fold the 4 groups
        if phone.transcription in GROUP_1:
            phone.transcription = 'axr/er'
            res['axr/er'].append(phone)
        elif phone.transcription in GROUP_2:
            phone.transcription = 'm/em'
            res['m/em'].append(phone)
        elif phone.transcription in GROUP_3:
            phone.transcription = 'n/en/nx'
            res['n/en/nx'].append(phone)
        elif phone.transcription in GROUP_4:
            phone.transcription = 'ng/eng'
            res['ng/eng'].append(phone)
        else:
            res[phone.transcription].append(phone)
    return res


def get_n_from_each_group(phone_groups: dict[str, list[Phone]],
                          n: int) -> list[Phone]:
    # BCM: An "advanced" Python trick is to use a "list constructor" here.
    #     return [random.sample(group, n) for group in phone_groups.values()]
    res = []
    for group in phone_groups.values():
        res += random.sample(group, n)
    return res


def predict_phone(train_set_phones: list[Phone], test_phone: Phone) -> str:

    # using KNN to find the nearest neighbor
    # BCM: k should be specified by a command-line argument, and be defaulted
    # to 10.
    k = 10
    # using a heap to keep track of the samllest k element
    # the items in the heap are tuples like (negative distance to the test_set_phone, train_set_phone transcription)
    # BCM: this needs more elaboration. Is heapq a min-heap or a max-heap?
    # How does the keeping track actually work? Pseudocode would be welcome here.
    heap = []
    heapq.heapify(heap)

    # BCM: Move this to the top of the function.
    if args.distance == "dtw":
        metric_distance = lambda p1, p2: p1.dtw_distance_to(p2)
    elif args.distance == "euclid":
        metric_distance = lambda p1, p2: p1.distance_to(p2)
    else:
        assert False, f"unknown distance metric: {args.distance}"

    # BCM: Using two variables that differ only by singular/plural
    # is considered bad style, as it can lead to easy errors.
    for train_set_phone in train_set_phones:
        distance = metric_distance(test_phone, train_set_phone)
        if len(heap) < k:
            heapq.heappush(heap, (-distance, train_set_phone.transcription))
        else:
            if -heap[0][0] > distance:
                heapq.heapreplace(heap,
                                  (-distance, train_set_phone.transcription))

    # using Counter to get the most common phone in the heap
    counter = Counter()
    for _ in range(k):
        _, transcription = heapq.heappop(heap)
        counter[transcription] += 1

    # predicted_phone is the most common phone in the heap
    predicted_phone = counter.most_common(1)[0][0]
    return predicted_phone


# BCM: Bad function name, since nothing is being tested
# here. Perhaps "trial"?
def test(train_set_phones: list[Phone], test_phones: list[Phone]):
    with open('test_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['True phone', 'Predicted phone'])
        correct_num = 0
        for test_phone in test_phones:
            print(f"Predicting {test_phone.transcription}...")
            predicted_phone = predict_phone(train_set_phones, test_phone)
            print(f"Predicted {predicted_phone}")
            if predicted_phone == test_phone.transcription:
                correct_num += 1
            writer.writerow([test_phone.transcription, predicted_phone])
        print(f"The accuracy is {correct_num / len(test_phones)}")


# stretch the phones to 1024 samples long
def stretch_phones(phones: list[Phone]):
    for phone in phones:
        phone.data = librosa.effects.time_stretch(
            phone.data,
            rate=(len(phone.data) / 1024),
            n_fft=512,
        )
        assert len(phone.data) == 1024, "incorrect phone resize"

def report_stats(phones):
    if args.verbose:
        phone_lens = [len(p.data) for p in phones]
        pls = [min(phone_lens), sum(phone_lens) / len(phone_lens), max(phone_lens)]
        print(f"phone lens: min={pls[0]} avg={pls[1]} max={pls[2]}")

if __name__ == "__main__":
    if args.stretch:
        namer = lambda t: f"stretched_{t}_set_phones.pkl"
        train_set_phones, test_set_phones, pkld = get_phones(namer, do_pkl=False)
    else:
        namer = lambda t: f"raw_{t}_set_phones.pkl"
        train_set_phones, test_set_phones, pkld = get_phones(namer, do_pkl=True)

    train_set_phones = drop_ignored_phones(train_set_phones)
    test_set_phones = drop_ignored_phones(test_set_phones)

    report_stats(train_set_phones + test_set_phones)

    if args.stretch and not pkld:
        stretch_phones(train_set_phones)
        stretch_phones(test_set_phones)
        for phone in train_set_phones + test_set_phones:
            phone.get_mfcc_seq()
        save_phones_to_pkl(train_set_phones, namer("train"))
        save_phones_to_pkl(test_set_phones, namer("test"))

    # BCM: Should we be using a random sample or a random
    # sample of instances from each phone?
    test_set = random.sample(test_set_phones, 1000)
    if args.stretch:
        stretch_phones(test_set)
    test(train_set_phones, test_set)
    # confusion matrix test
    # BCM: These labels are obtainable from the dictionary used in group_phones().
    labels = [
        'ix', 'iy', 's', 'r', 'n/en/nx', 'l', 'tcl', 'kcl', 'ih', 'dcl', 'k',
        't', 'm/em', 'eh', 'ae', 'axr/er', 'ax', 'z', 'd', 'q', 'w', 'ao',
        'aa', 'dh', 'pcl', 'p', 'dx', 'f', 'b', 'ah', 'ay', 'gcl', 'ey', 'sh',
        'ow', 'bcl', 'g', 'v', 'y', 'ux', 'ng/eng', 'jh', 'hv', 'hh', 'el',
        'th', 'oy', 'ch', 'uh', 'aw', 'uw', 'ax-h', 'zh'
    ]
