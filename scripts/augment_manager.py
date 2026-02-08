"""Batch augmentation manager: generate, review, analyze, iterate.

Usage examples:
  python scripts/augment_manager.py generate --n 100 --out aug_batch
  python scripts/augment_manager.py review --dir aug_batch
  python scripts/augment_manager.py analyze --dir aug_batch --out weights.json
  python scripts/augment_manager.py loop --n 100 --rounds 3 --out aug_batch

The reviewer opens images with the OS default image viewer and prompts for
feedback (y/n or 0-5 score). Feedback is saved as JSON next to images.
After a review pass, `analyze` computes per-augmentation scores and
produces adjusted sampling weights (decreases weight for augmentations with
low average score).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ensure repo root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.augment_250_run import (
    AUG_FNS,
    parse_svg_to_primitives,
    render,
    OUT_DIM,
    BASE_DIR,
)


def generate_batch(out_dir: Path, n: int, weights: Dict[str, float] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_files = list(BASE_DIR.glob('*.svg'))
    if len(svg_files) == 0:
        print('No SVGs found in', BASE_DIR)
        return

    chosen = random.sample(svg_files, min(n, len(svg_files)))

    aug_names = list(AUG_FNS.keys())
    # default uniform weights
    if weights is None:
        weights_list = None
    else:
        weights_list = [weights.get(name, 1.0) for name in aug_names]

    for idx, svg in enumerate(chosen):
        prim_sets = parse_svg_to_primitives(svg)
        if sum([0 if a.size == 0 else 1 for a in prim_sets.values()]) == 0:
            continue

        rendered = render(prim_sets, OUT_DIM, data_representation='vahe')
        if rendered.dtype == np.uint8:
            arrf = rendered.astype(np.float32) / 255.0
        else:
            arrf = rendered.astype(np.float32)
            if arrf.max() > 1.0:
                arrf = arrf / 255.0

        base_name = f'{idx:04d}'
        clean_path = out_dir / f'{base_name}_clean.png'
        from scripts.augment_250_run import save_uint8_gray

        save_uint8_gray(arrf, clean_path)

        num_augs = random.randint(1, 4)
        if weights_list is None:
            chosen_augs = random.sample(aug_names, num_augs)
        else:
            # sample without replacement respecting weights (approx)
            chosen_augs = []
            pool = aug_names.copy()
            pool_weights = weights_list.copy()
            for _ in range(num_augs):
                if not pool:
                    break
                pick = random.choices(pool, weights=pool_weights, k=1)[0]
                i = pool.index(pick)
                chosen_augs.append(pick)
                # remove picked
                pool.pop(i)
                pool_weights.pop(i)

        meta = {'svg': str(svg), 'base_name': base_name, 'augs': []}

        aug_img = arrf.copy()
        for a_name in chosen_augs:
            intensity = random.random()
            n_param = int(1 + intensity * 10)
            fn = AUG_FNS[a_name]
            try:
                aug_img = fn(aug_img, intensity)
            except Exception as e:
                print('  Aug failed', a_name, e)
                continue
            meta['augs'].append({'name': a_name, 'intensity': float(intensity), 'params': {'n': int(n_param)}})

        out_path = out_dir / f'{base_name}_aug.png'
        save_uint8_gray(aug_img, out_path)

        meta_path = out_dir / f'{base_name}_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)


def review_batch(dir_path: Path) -> None:
    """Open each augmented image and prompt user for feedback.

    Feedback format saved as JSON per-sample: {'score': int or None, 'note': str}
    Score: -1 = skip, 0-5 where 5 is excellent, 0 is unusable. 'y' maps to 4, 'n' to 0.
    """
    files = sorted(dir_path.glob('*_aug.png'))
    if not files:
        print('No augmented images found in', dir_path)
        return

    for p in files:
        meta_path = p.with_name(p.stem.replace('_aug', '_meta') + '.json')
        feedback_path = p.with_name(p.stem + '_feedback.json')
        # skip if already reviewed
        if feedback_path.exists():
            continue

        print('\nImage:', p)
        try:
            # platform-open
            if sys.platform.startswith('win'):
                os.startfile(p)
            else:
                # POSIX fallback
                opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                os.system(f"{opener} '{p}'")
        except Exception:
            pass

        # load meta for augmentation names
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        aug_list = [a.get('name') for a in meta.get('augs', [])]

        # prompt: supports single score or comma-separated bad aug indexes/names
        while True:
            resp = input(
                "Rate this image (y/n or 0-5), or list bad augs (e.g. 0,2 or name1,name2), s=skip, q=quit: "
            ).strip()
            if not resp:
                print('Invalid input')
                continue
            low = resp.lower()
            if low == 'q':
                return
            if low == 's':
                feedback = {'score': None, 'note': 'skipped', 'bad_augs': []}
                break
            # handle simple y/n
            if low == 'y':
                feedback = {'score': 4, 'note': '', 'bad_augs': []}
                break
            if low == 'n':
                feedback = {'score': 0, 'note': '', 'bad_augs': aug_list}
                break

            # comma-separated list -> try parse indexes first, then names
            if ',' in resp:
                parts = [p.strip() for p in resp.split(',') if p.strip()]
                bad_names = []
                for p in parts:
                    # try index
                    try:
                        i = int(p)
                        if 0 <= i < len(aug_list):
                            bad_names.append(aug_list[i])
                        else:
                            print(f'Index out of range: {i}')
                    except Exception:
                        # treat as name (exact match)
                        if p in aug_list:
                            bad_names.append(p)
                        else:
                            print(f'Unknown augmentation name: {p}')
                feedback = {'score': None, 'note': '', 'bad_augs': bad_names}
                break

            # single numeric score
            try:
                sc = int(resp)
                if 0 <= sc <= 5:
                    feedback = {'score': sc, 'note': '', 'bad_augs': []}
                    break
            except Exception:
                pass
            # single name -> mark that augmentation as bad
            if resp in aug_list:
                feedback = {'score': None, 'note': '', 'bad_augs': [resp]}
                break

            print('Invalid input')

        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=2)


def analyze_feedback(dir_path: Path) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Compute per-augmentation average scores and counts.

    Returns (avg_score_by_aug, count_by_aug)
    """
    aug_scores: Dict[str, List[float]] = defaultdict(list)
    aug_counts: Dict[str, int] = defaultdict(int)

    for meta_path in dir_path.glob('*_meta.json'):
        base = meta_path.stem.replace('_meta', '')
        aug_path = dir_path / f'{base}_aug.png'
        feedback_path = dir_path / f'{base}_aug_feedback.json'
        if not feedback_path.exists():
            # try alternate name without extra suffix
            feedback_path = dir_path / f'{base}_feedback.json'
        if not feedback_path.exists():
            continue
        with open(feedback_path, 'r', encoding='utf-8') as f:
            fb = json.load(f)
        score = fb.get('score')
        if score is None:
            continue
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        for a in meta.get('augs', []):
            name = a.get('name')
            aug_scores[name].append(float(score))
            aug_counts[name] += 1

    avg_scores: Dict[str, float] = {}
    for name, scores in aug_scores.items():
        avg_scores[name] = float(np.mean(scores)) if scores else 0.0

    return avg_scores, aug_counts


def adjust_weights(avg_scores: Dict[str, float], base_weights: Dict[str, float] | None = None) -> Dict[str, float]:
    """Produce adjusted weights: lower weight for aug with avg score < 3.

    Simple heuristic: new_weight = base_weight * (1 + (avg_score-3)/3)
    Clipped to [0.1, 5.0]
    """
    new_weights = {}
    for name in AUG_FNS.keys():
        base = 1.0 if base_weights is None else base_weights.get(name, 1.0)
        avg = avg_scores.get(name, 3.0)  # assume neutral if missing
        factor = 1.0 + (avg - 3.0) / 3.0
        w = float(base) * float(factor)
        w = max(0.1, min(5.0, w))
        new_weights[name] = w
    return new_weights


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')

    g = sub.add_parser('generate')
    g.add_argument('--n', type=int, default=100)
    g.add_argument('--out', type=str, default='aug_batch')
    g.add_argument('--weights', type=str, default=None, help='JSON file with weights')

    r = sub.add_parser('review')
    r.add_argument('--dir', type=str, required=True)

    a = sub.add_parser('analyze')
    a.add_argument('--dir', type=str, required=True)
    a.add_argument('--out', type=str, default='aug_weights.json')

    l = sub.add_parser('loop')
    l.add_argument('--n', type=int, default=100)
    l.add_argument('--rounds', type=int, default=3)
    l.add_argument('--out', type=str, default='aug_iter')

    args = p.parse_args()

    if args.cmd == 'generate':
        weights = None
        if args.weights:
            with open(args.weights, 'r', encoding='utf-8') as f:
                weights = json.load(f)
        generate_batch(Path(args.out), args.n, weights)

    elif args.cmd == 'review':
        review_batch(Path(args.dir))

    elif args.cmd == 'analyze':
        avg, counts = analyze_feedback(Path(args.dir))
        outp = Path(args.out)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump({'avg_scores': avg, 'counts': counts}, f, indent=2)
        print('Wrote analysis to', outp)

    elif args.cmd == 'loop':
        workdir = Path(args.out)
        base_weights = None
        for r_i in range(args.rounds):
            round_dir = workdir / f'round_{r_i:02d}'
            print(f'Generating round {r_i} ->', round_dir)
            generate_batch(round_dir, args.n, base_weights)
            print('Review the images now (you can run `review --dir {}`)'.format(round_dir))
            input('Press Enter when review is complete...')
            avg, counts = analyze_feedback(round_dir)
            base_weights = adjust_weights(avg, base_weights)
            # save weights
            weights_path = workdir / f'weights_round_{r_i:02d}.json'
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            with open(weights_path, 'w', encoding='utf-8') as f:
                json.dump(base_weights, f, indent=2)
            print('Saved weights to', weights_path)

    else:
        p.print_help()


if __name__ == '__main__':
    main()
