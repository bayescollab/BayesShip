#!/usr/bin/env python3
"""Merge the per-batch sample files of a BayesShip run into one HDF5 file.

A finished run already writes <moniker>_output.hdf5 itself. Use this on a run that is still going, was
stopped, or was killed: it concatenates the batches that the newest valid checkpoint generation refers to
(so it never picks up a half-finished batch) into the usual MCMC_OUTPUT layout.

    python merge_batches.py <outputDir> <moniker> [merged.hdf5]
"""
import glob
import json
import os
import sys

import h5py
import numpy as np


def latest_batches(out_dir, moniker):
    gens = sorted(glob.glob(os.path.join(out_dir, moniker + "_checkpoints", "gen_??????")), reverse=True)
    for g in gens:
        try:
            with open(os.path.join(g, "sampler.json")) as f:
                names = [b["name"] for b in json.load(f)["batchFiles"]]
        except (OSError, ValueError, KeyError):
            continue  # unreadable generation: try the previous one
        paths = [os.path.join(out_dir, moniker + "_batches", n) for n in names]
        if all(os.path.exists(p) for p in paths):
            return os.path.basename(g), paths
    raise SystemExit("no usable checkpoint generation found in " + out_dir)


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    out_dir, moniker = sys.argv[1], sys.argv[2]
    dest = sys.argv[3] if len(sys.argv) == 4 else os.path.join(out_dir, moniker + "_merged.hdf5")
    gen, paths = latest_batches(out_dir, moniker)
    if not paths:
        raise SystemExit("checkpoint " + gen + " refers to no sample batches yet")
    print("merging %d batches from %s" % (len(paths), gen))

    samples, logl = {}, {}
    for p in paths:
        with h5py.File(p, "r") as f:
            for k in f["MCMC_OUTPUT"]:
                if k.startswith("CHAIN"):
                    samples.setdefault(k, []).append(f["MCMC_OUTPUT"][k][:])
                    logl.setdefault(k, []).append(f["MCMC_OUTPUT/LOGL_LOGP"][k][:])
    tmp = dest + ".tmp"
    with h5py.File(paths[-1], "r") as last, h5py.File(tmp, "w") as out:
        out.create_group("MCMC_OUTPUT/LOGL_LOGP")
        for k in samples:
            out["MCMC_OUTPUT"].create_dataset(k, data=np.vstack(samples[k]), compression="gzip")
            out["MCMC_OUTPUT/LOGL_LOGP"].create_dataset(k, data=np.vstack(logl[k]), compression="gzip")
        last.copy("MCMC_METADATA", out)
    os.replace(tmp, dest)
    print("wrote", dest, {k: np.vstack(v).shape for k, v in samples.items()})


if __name__ == "__main__":
    main()
