data = dict(
    train=[
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/train/synth/MJ_train',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/train/synth/MJ_test',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/train/synth/MJ_valid',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/train/synth/ST',
        ),
        # dict(
        #     type='lmdb',
        #     name='Union14M-L-easy',
        #     data_root='./datasets/Union14M/Union14M-L/train_lmdb/train_easy',
        # ),
        # dict(
        #     type='lmdb',
        #     name='Union14M-L-medium',
        #     data_root='./datasets/Union14M/Union14M-L/train_lmdb/train_medium',
        # ),
        # dict(
        #     type='lmdb',
        #     name='Union14M-L-normal',
        #     data_root='./datasets/Union14M/Union14M-L/train_lmdb/train_normal',
        # ),
        # dict(
        #     type='lmdb',
        #     name='Union14M-L-hard',
        #     data_root='./datasets/Union14M/Union14M-L/train_lmdb/train_hard',
        # ),
        # dict(
        #     type='lmdb',
        #     name='Union14M-L-challenging',
        #     data_root='./datasets/Union14M/Union14M-L/train_lmdb/train_challenging',
        # ),
    ],
    train_unlabel=[
        dict(
            type='lmdb',
            data_root='./datasets/Union14M/Union14M-U/openvino_lmdb',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/Union14M/Union14M-U/cc_lmdb',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/Union14M/Union14M-U/boo32_lmdb',
        ),
    ],
    test=[
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/IIIT5k',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/SVT',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/IC13_857',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/IC15_1811',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/IC13_1015',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/IC15_2077',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/SVTP',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/CUTE80',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/Union14M/Union14M-L/Union14M-Benchmarks/lmdb_format/curve',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/Union14M/Union14M-L/Union14M-Benchmarks/lmdb_format/multi_oriented',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/Union14M/Union14M-L/Union14M-Benchmarks/lmdb_format/artistic',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/Union14M/Union14M-L/Union14M-Benchmarks/lmdb_format/contextless',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/Union14M/Union14M-L/Union14M-Benchmarks/lmdb_format/salient',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/Union14M/Union14M-L/Union14M-Benchmarks/lmdb_format/multi_words',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/Union14M/Union14M-L/Union14M-Benchmarks/lmdb_format/general',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/ArT',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/COCOv1.4',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/Uber',
        ),
        dict(
            type='lmdb',
            data_root='./datasets/rec_data/test/WordArt',
        ),
    ],
)
