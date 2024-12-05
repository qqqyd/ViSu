_base_ = ['./data_config.py']

charset = r"0123456789abcdefghijklmnopqrstuvwxyz"
data_shape = (32, 100)
max_length = 25

model = dict(
    max_length=max_length,
    charset_train=charset,
    charset_test=charset,
    pretrained=None,
    backbone=dict(
        img_size=data_shape,
        patch_size=(4, 8),
        in_chans=3, 
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        ),
    decode_head=dict(
        embed_dim=384,
        depth=1,
        num_heads=12,
        mlp_ratio=4,
        dropout=0.1,
    ),
    kl_thresh=0.5,
    conf_thresh=0.8,
    temperature=0.1,
    test_with_proj=False,
    loss_weight=dict(
        rec=1.0,
        simple_rec=1.0,
        cua=0.1,
        semi_cua=0.1,
        kl=1.0,
        semi_kl=1.0,
    ),
)

optimizer = dict(
    adjust_lr=True,
    lr=6e-4,
    betas=(0.9, 0.999),
    weight_decay=0.0,
    eps=1e-8,
)
scheduler = dict(
    type='OneCycle',
    warmup_pct=0.075,
)
workdir = './output/workdir/'

data = dict(
    samples_per_gpu=96,
    workers_per_gpu=16,
    charset_train=charset,
    charset_test=charset,
    data_shape=data_shape,
    max_length=max_length,
    remove_whitespace=True,
    normalize_unicode=True,
    rot2horizon=True,
    semi_train=True,
    ar_thresh=1.0,
    font_path='fonts',
    test_subset=('IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80', ),
    # test_subset=('artistic', 'contextless', 'curve', 'general', 'multi_oriented', 'multi_words', 'salient', ),
    # test_subset=('WordArt', 'ArT', 'COCOv1.4', 'Uber', ),
)

max_iter = 180000
write_log_interval = 1000
val_interval = 10000
