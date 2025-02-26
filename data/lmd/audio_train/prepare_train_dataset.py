import concurrent.futures as cf
import glob
import hashlib
import itertools
import json
import os
import pickle
import random
import re
import shutil
import sys

import librosa
import note_seq
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import pysndfx
import soundfile as sf
from note_seq.protobuf import music_pb2
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

INPUT_DIR = '/Users/ddulaev/PycharmProjects/ss-vq-kae/data/lmd/note_seq/olddata'
OUTPUT_DIR = 'wav_16kHz'
TOTAL_FILES = 169556
SR = 16000
SF_PATHS = {
    'train': [
        '../../soundfonts/fluid-soundfont-3.1/FluidR3_GM.sf2',
        '../../soundfonts/TimGM6mb.sf2',
        '../../soundfonts/Arachno SoundFont - Version 1.0.sf2'
    ],
    'val': [
        '../../soundfonts/fluid-soundfont-3.1/FluidR3_GM.sf2',
        '../../soundfonts/TimGM6mb.sf2',
        '../../soundfonts/Arachno SoundFont - Version 1.0.sf2'
    ],
    'test': [
        '../../soundfonts/TimbresOfHeaven/Timbres Of Heaven (XGM) 3.94.sf2'
    ]
}

# Load data augmentation parameters from metadata_ref.json instead of sampling them randomly.
# Set to True to reproduce the dataset from the paper. Set to False if you want to use your own data.
USE_REF_METADATA = True

if USE_REF_METADATA:
    with open('metadata_ref.json') as f:
        metadata_ref = json.load(f)
    metadata_ref_flat = {key: val for section in metadata_ref for key, val in metadata_ref[section].items()}

def filter_sequence(sequence, instrument_re=None, instrument_ids=None, programs=None, drums=None,
                    copy=False):
    if copy:
        sequence, original_sequence = music_pb2.NoteSequence(), sequence
        sequence.CopyFrom(original_sequence)

    if isinstance(instrument_re, str):
        instrument_re = re.compile(instrument_re)

    # Filter the instruments based on name and ID
    deleted_ids = set()
    if instrument_re is not None:
        deleted_ids.update(i.instrument for i in sequence.instrument_infos
                           if not instrument_re.search(i.name))
    if instrument_ids is not None:
        deleted_ids.update(i.instrument for i in sequence.instrument_infos
                           if i.instrument not in instrument_ids)
    new_infos = [i for i in sequence.instrument_infos if i.instrument not in deleted_ids]
    del sequence.instrument_infos[:]
    sequence.instrument_infos.extend(new_infos)

    # Filter the event collections
    for collection in [sequence.notes, sequence.pitch_bends, sequence.control_changes]:
        collection_copy = list(collection)
        del collection[:]

        for event in collection_copy:
            if event.instrument in deleted_ids:
                continue
            if instrument_ids is not None and event.instrument not in instrument_ids:
                continue
            if programs is not None and event.program not in programs:
                continue
            if drums is not None and event.is_drum != drums:
                continue
            collection.add().CopyFrom(event)

    return sequence

def random_fx(rng):
    chain = pysndfx.AudioEffectsChain()
    for _ in range(rng.choice([0, 1, 2, 3], p=[0.35, 0.4, 0.2, 0.05])):
        effect = rng.choice([
            lambda c: c.overdrive(gain=rng.uniform(10, 40)),
            lambda c: c.phaser(gain_in=rng.uniform(0.6, 0.9),
                               gain_out=rng.uniform(0.66, 0.85),
                               delay=rng.power(0.4) * 3 + 1,
                               decay=rng.uniform(0.2, 0.45),
                               speed=rng.uniform(0.5, 2),
                               triangular=rng.choice([True, False])),
            lambda c: c.gain(-3).reverb(),
            lambda c: c.tremolo(freq=rng.power(0.5) * 14 + 1,
                                depth=rng.uniform(20, 60))
        ])
        effect(chain)
    return chain


def process_file(args):
    path, sf_paths = args

    if USE_REF_METADATA:
        meta_key = os.path.splitext(os.path.basename(path))[0]
        meta_ref = metadata_ref_flat.get(meta_key)
        if meta_ref is None:
            return None
    else:
        # Use filename as seed
        seed = os.path.relpath(path, INPUT_DIR).encode()
        seed = int.from_bytes(hashlib.sha512(seed).digest(), 'big')
        rng = np.random.default_rng(seed=seed)

    with open(path, 'rb') as f:
        ns = pickle.load(f)
    if not ns.instrument_infos:
        return None
    max_instrument = max(ii.instrument for ii in ns.instrument_infos)

    meta = {
        'src_path': os.path.relpath(path, INPUT_DIR)
    }

    # Pick a random instrument
    if USE_REF_METADATA:
        instrument, program = meta_ref['instrument'], meta_ref['src_program']
    else:
        choices = sorted(set((n.instrument, n.program) for n in ns.notes if not n.is_drum))
        if not choices:
            return None
        instrument, program = choices[rng.choice(len(choices))]
    filter_sequence(ns, instrument_ids={instrument})
    meta['instrument'], meta['src_program'] = instrument, program

    # Change the program randomly
    if USE_REF_METADATA:
        program = meta_ref['program']
    else:
        if program < 32:  # Keyboards, guitars
            program = rng.choice(32)
        elif program >= 40 and program < 80:  # Strings, ensemble, brass, reed, pipe
            program = 40 + rng.choice(80 - 40)
        elif program < 104:
            # Pick a random program from the same class
            program = program - (program % 8) + rng.choice(8)
    meta['program'] = program

    for note in ns.notes:
        note.program = program

    # Pick a random SoundFont
    if USE_REF_METADATA:
        [sf_path] = [p for p in sf_paths if os.path.basename(p) == meta_ref['soundfont']]
    else:
        sf_path = rng.choice(sf_paths)
    meta['soundfont'] = os.path.basename(sf_path)

    # Pick two non-silent segments
    boundaries = np.arange(0., ns.total_time, 8.)
    if USE_REF_METADATA:
        indices = [s['index'] for s in meta_ref['segments']]
    else:
        onset_counts, _ = np.histogram([n.start_time for n in ns.notes], bins=boundaries)
        activity_map = (onset_counts >= 4)  # arbitrary threshold
        [candidates] = np.nonzero(activity_map)
        if len(candidates) < 2:
            return None
        indices = rng.choice(candidates, 2, replace=False)

    # Pick random effects
    if USE_REF_METADATA:
        effects = pysndfx.AudioEffectsChain()
        effects.command = meta_ref['effects']
    else:
        effects = random_fx(rng)
    meta['effects'] = effects.command

    meta['segments'] = []
    for ii, i in enumerate(indices):
        # Extract the chosen segment
        segment = note_seq.sequences_lib.extract_subsequence(ns, boundaries[i], boundaries[i + 1])

        # Transpose by a random amount (up to a fourth)
        if USE_REF_METADATA:
            assert i == meta_ref['segments'][ii]['index']
            transposition = meta_ref['segments'][ii]['transposition']
        else:
            transposition = rng.choice(np.arange(-5, 6))
        note_seq.sequences_lib.transpose_note_sequence(segment, transposition, in_place=True)

        # Synthesize it
        audio = note_seq.midi_synth.fluidsynth(segment, sf2_path=sf_path, sample_rate=SR)

        # Apply effects
        if len(audio) > 0:
            audio = effects(audio, sample_in=SR)

        # Clip to 8 seconds
        audio = audio[:8 * SR]

        instrument_len = len(str(max_instrument))
        i_len = len(str(len(boundaries) + 1))
        out_path = os.path.splitext(path)[0] + f'.{str(instrument).zfill(instrument_len)}.{str(i).zfill(i_len)}.wav'
        out_path = os.path.join(OUTPUT_DIR, os.path.relpath(out_path, INPUT_DIR))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, audio, SR, subtype='PCM_24')

        meta['segments'].append({
            'path': os.path.relpath(out_path, OUTPUT_DIR),
            'index': i,
            'transposition': transposition
        })

    return meta

def split():
    if USE_REF_METADATA:
        train_paths, val_paths, test_paths = (
            [os.path.join(INPUT_DIR, x['src_path']) for x in metadata_ref[k].values()]
            for k in ['train', 'val', 'test'])
        paths = [*train_paths, *val_paths, *test_paths]
    else:
        paths = list(tqdm(glob.iglob(os.path.join(INPUT_DIR, '**', '*.pickle'), recursive=True), desc='collect',
                          total=TOTAL_FILES))
        paths.sort()

        np.random.seed(42)
        np.random.shuffle(paths)
        train_paths, test_paths = train_test_split(paths, test_size=0.01)
        train_paths, val_paths = train_test_split(train_paths, test_size=800)

    metadata = {}
    with cf.ProcessPoolExecutor(16) as pool:
        for path_list, dset in [(train_paths, 'train'), (val_paths, 'val'), (test_paths, 'test')]:
            args = [(p, SF_PATHS[dset]) for p in path_list]
            metadata[dset] = list(tqdm(
                pool.map(process_file, args, chunksize=100),
                desc=f'convert {dset}', total=len(path_list)))

    metadata = {k: [p for p in metadata[k] if p is not None] for k in metadata}

    print(sum(len(m) for m in metadata.values()), '/', TOTAL_FILES, 'files converted successfully')

    return metadata

def prepare_train_dataset():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Директория {OUTPUT_DIR} успешно удалена.")
    else:
        print(f"Директория {OUTPUT_DIR} не существует.")

    os.makedirs(OUTPUT_DIR)

    metadata = split()

    metadata = {k: {os.path.splitext(os.path.basename(m['src_path']))[0]: m for m in metadata[k]}
                for k in metadata}

    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, cls=NumPyJSONEncoder)

    def write_tuples(tuples, path, shuffle_items=False):
        with open(path, 'w') as f:
            for tup in tuples:
                if shuffle_items:
                    tup = np.random.choice(tup, size=len(tup), replace=False)
                print(*tup, sep='\t', file=f)

    np.random.seed(42)
    for dset in ['train', 'val', 'test']:
        path_pairs = [(os.path.join(OUTPUT_DIR, a['path']), os.path.join(OUTPUT_DIR, b['path']))
                      for m in metadata[dset].values() for a, b in [m['segments']]]
        path_pairs.sort()
        np.random.shuffle(path_pairs)
        write_tuples(path_pairs, f'pairs_{dset}', shuffle_items=True)


class NumPyJSONEncoder(json.JSONEncoder):
    def default(self, x):
        if isinstance(x, (np.ndarray, np.generic)):
            return x.tolist()
        else:
            return super().default(x)

def main():
    prepare_train_dataset()

if __name__ == '__main__':
    main()