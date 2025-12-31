#!/bin/bash
set -e

# vLLM 빌드 필수 패키지 설치
apt-get update && apt-get install -y ninja-build cmake pkg-config

# NCCL 환경 변수 설정 (T4 멀티 GPU 안정성)
echo 'export NCCL_DISABLE_CHECK=1' >> /etc/environment
echo 'export NCCL_P2P_DISABLE=1' >> /etc/environment
