# Build stage
FROM rocm/dev-ubuntu-22.04:latest as builder

RUN apt-get update && apt-get install -y \
    python3 python3-pip git rocm-dev rocm-libs hipcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY src/ ./src/
COPY sample_kernels/ ./sample_kernels/

# Runtime stage
FROM rocm/dev-ubuntu-22.04:runtime

RUN apt-get update && apt-get install -y \
    python3 python3-pip rocm-dev rocm-libs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

ENTRYPOINT ["python3", "-m", "src.main"]
CMD ["--input", "sample_kernels/cuda/warp_reduce.cu"]
