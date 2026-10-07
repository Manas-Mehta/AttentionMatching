# KVzip as published (snu-mllab/KVzip @ 5d84729): context-dependent, pair-level eviction,
# non-uniform (global) budget, 2,000-token reconstruction chunks.
# Implementation: compaction/compaction_methods/kvzip.py. Run with --chunking none.
# (kvzip.py / kvzip-uniform.py in this folder are the AM authors' approximations, not this.)
config = {
    'kvzip_original': {
        'algorithm': 'kvzip',
    },
}
