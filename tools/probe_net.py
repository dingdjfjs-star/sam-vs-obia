# -*- coding: utf-8 -*-
"""网络通道探测：PyTorch CUDA 轮子源 / PyPI 镜像 / HuggingFace 镜像。

用法：E:\\Ana\\python.exe tools\\probe_net.py
"""
import concurrent.futures as cf
import urllib.request

CHECKS = {
    "pytorch_official_cu124": "https://download.pytorch.org/whl/cu124/torch/",
    "aliyun_pytorch_wheels": "https://mirrors.aliyun.com/pytorch-wheels/cu124/",
    "tuna_pypi": "https://pypi.tuna.tsinghua.edu.cn/simple/",
    "hf_co": "https://huggingface.co",
    "hf_mirror_api": "https://hf-mirror.com/api/models/facebook/sam2-hiera-small",
    "github_api": "https://api.github.com",
}


def probe(item):
    key, url = item
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sam-vs-obia-probe"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            return key, resp.status
    except Exception as e:  # noqa: BLE001
        return key, "ERR:" + type(e).__name__


if __name__ == "__main__":
    with cf.ThreadPoolExecutor(4) as ex:
        for key, status in ex.map(probe, CHECKS.items()):
            print(f"{key:24s} {status}", flush=True)
