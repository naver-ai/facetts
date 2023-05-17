import os
from sacred import Experiment

from model.utils import fix_len_compatibility

ex = Experiment("face-tts")


@ex.config
def config():
    seed = int(os.getenv("seed", 37))

    # Dataset Configs
    dataset = os.getenv("dataset", "lrs3")
    lrs3_train = os.getenv("lrs3_train", "datalist/lrs3_train_long.list")
    lrs3_val = os.getenv("lrs3_val", "datalist/lrs3_val_long.list")
    lrs3_test = os.getenv("lrs3_test", "datalist/lrs3_test_long.list")
    lrs3_path = os.getenv("lrs3_path", "data/lrs3")
    cmudict_path = os.getenv("cmudict_path", "utils/cmu_dictionary")

    # Data Configs
    image_size = int(os.getenv("image_size", 224))
    max_frames = int(os.getenv("max_frames", 30))
    image_augment = int(os.getenv("image_augment", 0))

    ## hifigan-16k setting
    n_fft = int(os.getenv("n_fft", 1024))
    sample_rate = int(os.getenv("sample_rate", 16000))
    hop_len = int(os.getenv("hop_len", 160))
    win_len = int(os.getenv("win_len", 1024))
    f_min = float(os.getenv("f_min", 0.0))
    f_max = float(os.getenv("f_max", 8000))
    n_mels = int(os.getenv("n_mels", 128))


    # Experiment Configs
    batch_size = int(os.getenv("batch_size", 256))
    add_blank = int(os.getenv("add_blank", 1))  # True: 1 / False: 0
    snet_emb = int(os.getenv("snet_emb", 1))  # True: 1 / False: 0
    n_spks = int(os.getenv("n_spks", 2007))  # libritts:247, lrs3: 2007
    multi_spks = int(os.getenv("multi_spks", 1))
    out_size = fix_len_compatibility(2 * sample_rate // 256)
    model = os.getenv("model", "face-tts")

    # Network Configs

    ## Encoder parameters
    n_feats = n_mels
    spk_emb_dim = int(os.getenv("spk_emb_dim", 64))  # For multispeaker Grad-TTS
    vid_emb_dim = int(os.getenv("vid_emb_dim", 512))  # For Face-TTS
    n_enc_channels = int(os.getenv("n_enc_channels", 192))
    filter_channels = int(os.getenv("filter_channels", 768))
    filter_channels_dp = int(os.getenv("filter_channels_dp", 256))
    n_enc_layers = int(os.getenv("n_enc_layers", 6))
    enc_kernel = int(os.getenv("enc_kernel", 3))
    enc_dropout = float(os.getenv("enc_dropout", 0.0))
    n_heads = int(os.getenv("n_heads", 2))
    window_size = int(os.getenv("window_size", 4))

    ## Decoder parameters
    dec_dim = int(os.getenv("dec_dim", 64))
    beta_min = float(os.getenv("beta_min", 0.05))
    beta_max = float(os.getenv("beta_max", 20.0))
    pe_scale = float(os.getenv("pe_scale", 1000.0))

    ## Syncnet parameters
    syncnet_stride = int(os.getenv("syncnet_stride", 1))
    syncnet_ckpt = os.getenv("syncnet_ckpt")
    spk_emb = os.getenv("spk_emb", "face")

    # Optimizer Configs
    optim_type = os.getenv("optim_type", "adam")
    schedule_type = os.getenv("schedule_type", "constant")
    learning_rate = float(os.getenv("learning_rate", 1e-4))
    end_lr = float(os.getenv("end_lr", 1e-7))
    weight_decay = float(os.getenv("weight_decay", 0.1))
    decay_power = float(os.getenv("decay_power", 1.0))
    max_steps = int(os.getenv("max_steps", 100000))

    save_step = int(os.getenv("save_step", 10000))
    warmup_steps = float(os.getenv("warmup_steps", 0))  # 1000

    video_data_root = os.getenv("video_data_root", "mp4")
    image_data_root = os.getenv("image_data_root", "jpg")
    audio_data_root = os.getenv("audio_data_root", "wav")

    log_dir = os.getenv("CHECKPOINTS", "./logs")
    log_every_n_steps = int(os.getenv("log_every_n_steps", 1000))

    num_gpus = int(os.getenv("num_gpus", 1))
    per_gpu_batchsize = int(batch_size / num_gpus)
    num_nodes = int(os.getenv("num_nodes", 1))
    num_workers = int(os.getenv("num_workers", 2))
    prefetch_factor = int(os.getenv("prefetch_factor", 2))

    # Inference Configs
    test_txt = os.getenv("test_txt", "test/text.txt")
    use_custom = int(os.getenv("use_custom", 1))
    test_faceimg = os.getenv("test_faceimg", "test/face2.png")
    timesteps = int(os.getenv("timesteps", 10))
    output_dir = os.getenv("output_dir", "test")

    # SyncNet Configs
    syncnet_initw = float(os.getenv("syncnet_initw", 10.0))
    syncnet_initb = float(os.getenv("syncnet_initb", -5.0))

    resume_from = os.getenv("resume_from", "./ckpts/facetts_lrs3.pt")
    val_check_interval = float(os.getenv("val_check_interval", 1.0))
    test_only = int(os.getenv("test_only", 0))
