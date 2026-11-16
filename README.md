English | <a href="README.zh-CN.md">简体中文</a>

<p align="center"><img src="frontend/public/laya-server-logo-black.png" alt="LAYA SERVER" width="120" /></p>

<h1 align="center">LAYA SERVER</h1>

<p align="center">
  <a href="https://github.com/1Panel-dev/laya-server/releases"><img src="https://img.shields.io/github/v/release/1Panel-dev/laya-server" alt="Release"></a>
  <a href="https://github.com/1Panel-dev/laya-server/stargazers"><img src="https://img.shields.io/github/stars/1Panel-dev/laya-server" alt="GitHub stars"></a>
  <a href="https://hub.docker.com/r/1panel/laya-server"><img src="https://img.shields.io/docker/pulls/1panel/laya-server" alt="Docker pulls"></a>
  <a href="https://laya-ai.com/projects#project-1panel-dev-laya-server"><img src="https://laya-ai.com/badges/listed-on-laya-ai.svg" alt="Listed on laya-ai.com"></a>
</p>

LAYA SERVER is a self-hosted API and web interface for Laya's structured decision models, compatible with the TypeSafe Jev API format. It packages the System One inference capabilities of [upstream Laya](https://github.com/NandhaKishorM/laya) as a standalone service, making them easier to deploy and use in your applications.

## Quick Start

Once the image is published, install Docker and run the following command. Replace `latest` with a version tag to pin a release.

```sh
docker run -d --name laya-server --init --restart unless-stopped --gpus all \
  -p 8080:8080 \
  -v "$(pwd)/data:/data" \
  -e LAYA_ADMIN_USERNAME=admin \
  -e LAYA_ADMIN_PASSWORD='change-this-admin-password' \
  -e LAYA_DEVICE=cuda \
  1panel/laya-server:latest
```

Replace the sample password with your own, then sign in at `http://YOUR_SERVER_IP:8080`. Create a key on the **API Keys** page; the full key is shown only once. `/health/ready` checks CUDA availability and the model files. Data is stored in the local `./data` directory, which is ignored by Git.

## GPU deployment

Model files are downloaded from [ModelScope](https://www.modelscope.cn/models/convaiinnovations/laya-multilingual)
with download integrity checked by the ModelScope SDK using repository metadata. Only the required weights, tokenizer and
encoder configuration are included. Runtime inference stays offline.
To download locally: `pip install modelscope==1.34.0`, then run
`python scripts/download-models.py`. The script only downloads `multilingual` to
`./models/multilingual` by default. Override the directory with `LAYA_MODEL_DIR`.
Compose mounts `./models` at `/opt/models`; download the model before starting Compose.
The `models/` and `data/` directories are ignored by Git. Set `LAYA_PORT` in `.env`
to change the host port (default: `8080`).

The image installs PyTorch 2.5.1 with CUDA 12.4 and defaults to `LAYA_DEVICE=cuda`.
The host needs a compatible NVIDIA GPU and driver, plus the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
GPU access follows the [Docker Compose GPU configuration](https://docs.docker.com/compose/how-tos/gpu-support/).
Startup fails with a diagnostic if CUDA is unavailable. Set `LAYA_DEVICE=cuda:1` to
select the second GPU visible inside the container; a single process uses one GPU.
Upstream Laya retains its CPU retry behavior for GPU memory errors, so size request
lengths and concurrency to fit your GPU memory.

To build this checkout locally before a GPU image is published:

```sh
git clone --depth 1 --branch v0.4.2 https://github.com/NandhaKishorM/laya.git laya
git -C laya checkout --detach 68804629e8ccd9d616d48a40e87de9fabbeae069
docker build -t 1panel/laya-server:gpu-local .
# Copy .env.example to .env and fill in the admin password first.
LAYA_IMAGE_TAG=gpu-local docker compose up -d
docker compose exec app python /app/scripts/smoke-image-model.py
```

If `laya/` already exists, use that checkout at the pinned commit instead of cloning
again. Builds run the bundled model check on CPU and do not need a GPU. The final
command runs real offline inference on the configured GPU and checks model placement.
For explicit CPU operation, set `LAYA_DEVICE=cpu` and omit `--gpus all`, or remove
`deploy.resources.reservations.devices` from `compose.yaml`.

`YOUR_SERVER_IP` is your server's IP address. To use a domain name, configure HTTPS on a reverse proxy and forward requests to port 8080 on the server.

## Screenshot

![LAYA SERVER Playground](docs/images/screenshot-en.png)

## Call the API

```sh
curl -X POST http://YOUR_SERVER_IP:8080/v1/systemone \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"state":{"message":"I was charged twice"},"questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}'
```

Successful responses include `answers`, `model`, and actual token counts in `usage`. The published image includes the multilingual model; `model=auto` (the default) and `model=multilingual` are available. See **Documentation** in the console for more request examples.

## Usage statistics and storage

Inference responses enqueue usage statistics without waiting for SQLite writes. A single
background thread per process reuses a connection, writes batches of up to 100 events,
and coalesces `last_used_at` updates by API key. Partial batches flush after up to one
second when the writer is idle; slow storage can delay statistics further. The `usage`
field in each inference response is unchanged. `duration_ms` measures model prediction
and result validation, not queue time or database writes.

Statistics are best-effort, eventually consistent telemetry, not a billing or quota
ledger. Each process buffers up to 1,000 queued events plus one in-flight batch. A full
queue drops new statistics with rate-limited warnings. SQLite failures roll back where
possible, log and drop the affected batch, then reconnect for later batches; failed
batches are not retried because an uncertain commit could double-count usage. Neither
condition turns successful inference into an HTTP 503. API-key authentication and
revocation still use the database synchronously.

Graceful shutdown attempts to drain the queue for up to five seconds. An in-flight
SQLite call cannot be cancelled safely and may outlive that wait; forced termination,
a shutdown timeout, or storage errors can lose unflushed statistics. Keep `/data` on
low-latency persistent storage: background batching reduces write pressure but does
not fix slow authentication reads or database locking.

## License and Feedback

This project is licensed under [Apache License 2.0](LICENSE). For issues and suggestions, visit [GitHub Issues](https://github.com/1Panel-dev/laya-server/issues).
