# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# HOMIE-Wan subject-to-video generation entry point (NVIDIA GPU).
#
# Mirrors Phantom/generate.py but drives the HOMIE r2v pipeline and reads the HOMIE
# meta-file format (reference_paths / prompt / qwen_feature), as produced for the NPU
# run (eval_datasets/nips_new_100.jsonl). Supports single-GPU and FSDP multi-GPU.
import argparse
import logging
import os
import random
import resource
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist

import homie_wan
from homie_wan.configs import HOMIE_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, get_config
from homie_wan.utils.utils import cache_video, load_homie_jsonl, load_image, str2bool


def _rss_gib() -> float:
    # Linux: ru_maxrss is KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _log_memory(tag: str, device: int = 0):
    msg = f"[mem {tag}] peak RSS={_rss_gib():.2f} GiB"
    if torch.cuda.is_available() and device < torch.cuda.device_count():
        msg += (
            f" CUDA[{device}]={torch.cuda.get_device_name(device)}"
            f" allocated={torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GiB"
            f" reserved={torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GiB"
        )
    logging.info(msg)


def _select_cuda_device(local_rank: int) -> int:
    """Bind a usable CUDA device; fail clearly if none are compatible."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA device is available inside the container.")
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested CUDA device {local_rank} but only "
            f"{torch.cuda.device_count()} device(s) are visible. "
            "Check CUDA_VISIBLE_DEVICES / nvidia-smi ordering."
        )
    torch.cuda.set_device(local_rank)
    # Touch the device so incompatible arches fail before model load.
    torch.zeros(1, device=f"cuda:{local_rank}")
    props = torch.cuda.get_device_properties(local_rank)
    logging.info(
        "Using CUDA device %d: %s (sm_%d%d, %.1f GiB)",
        local_rank,
        torch.cuda.get_device_name(local_rank),
        props.major,
        props.minor,
        props.total_memory / (1024**3),
    )
    torch.cuda.reset_peak_memory_stats(local_rank)
    return local_rank


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify --ckpt_dir (HF Wan2.1-T2V-14B-Diffusers)."
    assert args.homie_ckpt is not None, "Please specify --homie_ckpt (HOMIE-Wan-Model dir)."
    assert args.task in HOMIE_CONFIGS, f"Unsupported task: {args.task}"

    if args.sample_steps is None:
        args.sample_steps = 50
    if args.sample_shift is None:
        args.sample_shift = 3.0
    if args.frame_num is None:
        args.frame_num = 97

    if args.quantized_dit and args.homie_ckpt_basename == "Homie_Wan_14B":
        args.homie_ckpt_basename = "Homie_Wan_14B_q8"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupported size {args.size} for task {args.task}, supported: "
        f"{', '.join(SUPPORTED_SIZES[args.task])}"
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a subject-consistent video with HOMIE-Wan (NVIDIA GPU)."
    )
    parser.add_argument("--task", type=str, default="s2v-14B", choices=list(HOMIE_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()),
                        help="Generated video resolution (width*height).")
    parser.add_argument("--frame_num", type=int, default=None, help="Number of frames (4n+1).")
    parser.add_argument("--sample_fps", type=int, default=24, help="FPS of the saved video.")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="HF Wan2.1-T2V-14B-Diffusers dir (vae/text_encoder/tokenizer).")
    parser.add_argument("--homie_ckpt", type=str, default=None,
                        help="Dir with Homie_Wan_14B*.safetensors + index json.")
    parser.add_argument("--homie_ckpt_basename", type=str, default="Homie_Wan_14B")
    parser.add_argument(
        "--quantized_dit",
        action="store_true",
        default=False,
        help="Load a Quanto qint8 DiT from --homie_ckpt (expects Homie_Wan_14B_q8 + quantization_map.json).",
    )
    parser.add_argument("--offload_model", type=str2bool, default=None,
                        help="Offload the DiT to CPU after denoising to save VRAM.")

    # multi-GPU
    parser.add_argument("--ulysses_size", type=int, default=1,
                        help="Ulysses context-parallel degree (only 1 is supported; see README).")
    parser.add_argument("--ring_size", type=int, default=1,
                        help="Ring attention degree (only 1 is supported).")
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)

    # inputs
    parser.add_argument("--input_json", type=str, default=None,
                        help="HOMIE meta jsonl (reference_paths/prompt/qwen_feature).")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single-sample prompt (used when --input_json is not given).")
    parser.add_argument("--ref_image", type=str, default=None,
                        help="Comma-separated reference image paths for a single sample.")
    parser.add_argument("--qwen_feature", type=str, default=None,
                        help="Path to a single-sample qwen feature .pt (optional).")

    # sampling
    parser.add_argument("--save_path", type=str, default="./homie_results",
                        help="Directory to write generated videos into.")
    parser.add_argument("--save_file", type=str, default=None,
                        help="Explicit output file (single-sample mode only).")
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None,
                        help="Flow-matching shift (flow_shift).")
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)

    args = parser.parse_args()
    _validate_args(args)
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format="[%(asctime)s] %(levelname)s: %(message)s",
                            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def _build_samples(args):
    """Return aligned lists (prompts, images, qwen_features, reference_id_labels)."""
    if args.input_json:
        return load_homie_jsonl(args.input_json)

    # single-sample mode
    assert args.prompt is not None and args.ref_image is not None, (
        "Without --input_json you must pass both --prompt and --ref_image."
    )
    ref_paths = [p for p in args.ref_image.split(",") if p]
    images = [[load_image(p) for p in ref_paths]]
    prompts = [args.prompt]
    if args.qwen_feature:
        from homie_wan.utils.utils import truncate_qwen_feature
        qwen = torch.load(args.qwen_feature, map_location="cpu")
        qwen_features = [truncate_qwen_feature(qwen)]
    else:
        qwen_features = [None]
    # one reference-id letter per reference image
    reference_id_labels = [[[chr(ord("a") + i)] for i in range(len(ref_paths))]]
    return prompts, images, qwen_features, reference_id_labels


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True

    if world_size > 1:
        device = _select_cuda_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), \
            "t5_fsdp/dit_fsdp require a distributed (torchrun) launch."
        device = _select_cuda_device(local_rank)

    if args.quantized_dit and (args.dit_fsdp or args.t5_fsdp or world_size > 1):
        raise ValueError("--quantized_dit is single-GPU only (no FSDP / multi-process).")

    assert args.ulysses_size == 1 and args.ring_size == 1, (
        "Context parallel (--ulysses_size/--ring_size > 1) is not implemented yet for "
        "HOMIE-Wan. Use FSDP (--dit_fsdp/--t5_fsdp) for multi-GPU. "
        "See homie_wan/distributed/context_parallel.py."
    )

    cfg = get_config(args.task)
    if args.sample_fps is not None:
        cfg.sample_fps = args.sample_fps

    logging.info(f"Generation job args: {args}")
    _log_memory("start", device)

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    prompts, images, qwen_features, reference_id_labels = _build_samples(args)
    num_real = len(prompts)
    logging.info(f"Loaded {num_real} sample(s).")

    # Pad the tail up to a multiple of world_size by duplicating the last sample, so every
    # rank gets the same number of samples. This keeps the per-rank FSDP forward counts
    # aligned (unequal counts would deadlock the all-gather). The padded (duplicate) videos
    # are still written; we warn about them at the end.
    padded_indices = []
    if world_size > 1 and (num_real % world_size) != 0:
        pad = world_size - (num_real % world_size)
        for _ in range(pad):
            padded_indices.append(len(prompts))
            prompts.append(prompts[num_real - 1])
            images.append(images[num_real - 1])
            qwen_features.append(qwen_features[num_real - 1])
            reference_id_labels.append(reference_id_labels[num_real - 1])
        logging.info(
            f"Padded {num_real} -> {len(prompts)} samples (x{world_size}); "
            f"indices {padded_indices} are duplicates of sample {num_real - 1}."
        )

    logging.info("Creating HOMIE-Wan S2V pipeline.")
    pipeline = homie_wan.HomieWanS2V(
        config=cfg,
        ckpt_dir=args.ckpt_dir,
        homie_ckpt=args.homie_ckpt,
        homie_ckpt_basename=args.homie_ckpt_basename,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=False,
        t5_cpu=args.t5_cpu,
        quantized_dit=args.quantized_dit,
    )
    _log_memory("pipeline_loaded", device)

    os.makedirs(args.save_path, exist_ok=True)

    # Distribute samples round-robin across ranks (data parallel over samples).
    for i in range(len(prompts)):
        if world_size > 1 and (i % world_size) != rank:
            continue

        save_file = args.save_file if (args.save_file and len(prompts) == 1) \
            else os.path.join(args.save_path, f"video_{i}.mp4")
        if os.path.exists(save_file):
            logging.info(f"{save_file} exists, skip.")
            continue

        logging.info(f"[sample {i}] generating: {prompts[i][:80]}...")
        video = pipeline.generate(
            input_prompt=prompts[i],
            ref_images=images[i],
            qwen_feature=qwen_features[i],
            reference_id_labels=reference_id_labels[i],
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

        if video is not None:  # rank 0 (or single GPU)
            cache_video(video[None], save_file=save_file, fps=cfg.sample_fps,
                        nrow=1, normalize=True, value_range=(-1, 1))
            logging.info(f"[sample {i}] saved to {save_file}")
            _log_memory(f"sample_{i}_done", device)

    if dist.is_initialized():
        dist.barrier()

    if padded_indices:
        dup_files = [os.path.join(args.save_path, f"video_{i}.mp4") for i in padded_indices]
        logging.warning(
            f"The following {len(padded_indices)} file(s) are padding duplicates of "
            f"sample {num_real - 1} (added to reach a multiple of world_size={world_size}) "
            f"and can be deleted: {dup_files}"
        )
    _log_memory("finished", device)
    logging.info("Finished.")


if __name__ == "__main__":
    generate(_parse_args())
