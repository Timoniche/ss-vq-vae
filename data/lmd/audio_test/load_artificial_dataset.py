import concurrent.futures as cf
import os
import pickle
import datetime
import warnings

import note_seq
import pretty_midi
from tqdm.auto import tqdm

# INPUT_DIR = '../note_seq/data/'
INPUT_DIR = '/Users/ddulaev/PycharmProjects/ss-vq-kae/data/lmd/lmd_mini_full'
OUTPUT_DIR = 'data'
TOTAL_FILES = 178561  # https://colinraffel.com/projects/lmd/

# Decrease MAX_TICK value to avoid running out of RAM. Long files will be skipped
pretty_midi.pretty_midi.MAX_TICK = 1e6

# Run ../note_seq/prepare.ipynb and ../audio_train/prepare.ipynb first.
def get_paths():
    for dirpath, _, filenames in os.walk(INPUT_DIR):
        for filename in filenames:
            yield os.path.join(dirpath, filename)


def process_file(path):
    if os.stat(path).st_size > 100000:
        return None, 0

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Tempo, Key or Time signature change events found on non-zero tracks')
            ns = note_seq.midi_io.midi_file_to_note_sequence(path)
    except (note_seq.midi_io.MIDIConversionError, Exception) as e:
        print(f"Skipping file {path} due to error: {e}")
        return None, 0
    out_path = os.path.splitext(path)[0] + f'.pickle'
    out_path = os.path.join(OUTPUT_DIR, os.path.relpath(out_path, INPUT_DIR))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(ns, f)
    return out_path, ns.total_time


def load_artificial_dataset():
    os.makedirs(OUTPUT_DIR)
    with cf.ProcessPoolExecutor(20) as pool:
        results = list(tqdm(
            pool.map(process_file, tqdm(get_paths(), desc='collect', total=TOTAL_FILES), chunksize=100),
            desc='convert', total=TOTAL_FILES))

    print(sum(1 for p, _ in results if p is not None), '/', len(results), 'files converted successfully')
    print('Total time:', datetime.timedelta(seconds=sum(t for _, t in results)))


def main():
    load_artificial_dataset()


if __name__ == '__main__':
    main()
