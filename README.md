# Генерація 1000 зображень з ComfyUI + FLUX

Відтворюваний опис експерименту: локальний запуск ComfyUI як API-сервісу з моделлю FLUX (GGUF Q4) на одній RTX 3060 (12 ГБ) та пакетна генерація 1000 фотореалістичних сцен через Python-скрипт.

## Апаратні та системні вимоги

- GPU: NVIDIA RTX 3060, 12 ГБ VRAM (compute capability 8.6)
- ОЗП: щонайменше 32 ГБ (для offloading вагів та pinned memory)
- ОС: Ubuntu (Linux x86_64)
- Драйвери NVIDIA з підтримкою CUDA 12.4
- Python 3.12, менеджер пакетів `uv`
- Вільне місце на диску: ~20 ГБ під моделі

## 1. Встановлення ComfyUI у venv (uv)

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
uv venv --python 3.12
source .venv/bin/activate

# PyTorch зі збіркою під CUDA 12.4
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

# Вузол-завантажувач GGUF (потрібен для Q4 FLUX)
cd custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF.git
cd ComfyUI-GGUF && uv pip install -r requirements.txt
cd ../..
```

### Типова проблема: torchaudio

Якщо при старті виникає помилка `OSError: libcudart.so.13: cannot open shared object file`, це означає невідповідність версії `torchaudio` до версії `torch`. Виправлення:

```bash
uv pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

ComfyUI для генерації зображень `torchaudio` не потребує — за потреби його можна просто видалити (`uv pip uninstall torchaudio`).

## 2. Завантаження моделей

```bash
uv pip install huggingface_hub

# Модель FLUX Q4 -> models/unet
hf download city96/FLUX.1-dev-gguf flux1-dev-Q4_K_S.gguf --local-dir models/unet

# Текстовий енкодер T5 Q4 -> models/clip
hf download city96/t5-v1_1-xxl-encoder-gguf t5-v1_1-xxl-encoder-Q4_K_S.gguf --local-dir models/clip

# clip_l -> models/clip
hf download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/clip

# VAE -> models/vae
hf download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir models/vae
```

> FLUX.1-dev — модель із обмеженим доступом (gated). Спершу прийміть ліцензію на сторінці моделі в Hugging Face та виконайте `hf auth login`. Альтернатива без обмежень — модель `schnell`.

Перевірка розташування файлів:

```bash
ls models/unet/   # flux1-dev-Q4_K_S.gguf
ls models/clip/   # t5-v1_1-xxl-encoder-Q4_K_S.gguf, clip_l.safetensors
ls models/vae/    # ae.safetensors
```

## 3. Запуск ComfyUI як systemd-сервісу

Файл `/etc/systemd/system/comfyui.service` (підставте свого користувача та шляхи):

```ini
[Unit]
Description=ComfyUI API server
After=network.target

[Service]
Type=simple
User=automation
Group=automation
WorkingDirectory=/home/automation/ComfyUI
ExecStart=/home/automation/ComfyUI/.venv/bin/python main.py --listen 127.0.0.1 --port 8188 --cache-lru 1
Restart=on-failure
RestartSec=5
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[Install]
WantedBy=multi-user.target
```

Встановлення та запуск:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now comfyui
systemctl status comfyui
journalctl -u comfyui -f          # живі логи
curl http://127.0.0.1:8188/system_stats
```

### Примітка про VRAM та перезавантаження моделі

- Прапорець `--lowvram` вивантажує ваги моделі в ОЗП між запусками, через що FLUX **перезавантажується щоразу**. На 12 ГБ модель Q4 (≈6.6 ГБ) вміщується й без нього.
- Для збереження моделі в пам'яті між запитами використовуйте звичайний режим VRAM + `--cache-lru 1` (як у юніті вище).
- Якщо без `--lowvram` виникає OOM на етапі VAE decode — додайте `--reserve-vram 0.5`.
- ComfyUI займає GPU одразу при старті. Перед запуском vLLM/Ollama на тій самій карті зупиніть сервіс: `sudo systemctl stop comfyui`.

## 4. Файл воркфлоу (API-формат)

ComfyUI приймає через API повний граф воркфлоу у форматі JSON — не лише назву моделі та промпт. Граф `workflow_api.json` для FLUX GGUF (text-to-image, без PuLID):

```json
{
  "1": { "class_type": "UnetLoaderGGUF",
         "inputs": { "unet_name": "flux1-dev-Q4_K_S.gguf" } },
  "2": { "class_type": "DualCLIPLoaderGGUF",
         "inputs": { "clip_name1": "t5-v1_1-xxl-encoder-Q4_K_S.gguf",
                     "clip_name2": "clip_l.safetensors", "type": "flux" } },
  "3": { "class_type": "VAELoader",
         "inputs": { "vae_name": "ae.safetensors" } },
  "9": { "class_type": "CLIPTextEncode",
         "inputs": { "clip": ["2", 0],
                     "text": "a man standing in a city street, photorealistic" } },
  "11": { "class_type": "FluxGuidance",
          "inputs": { "conditioning": ["9", 0], "guidance": 3.5 } },
  "12": { "class_type": "EmptyLatentImage",
          "inputs": { "width": 896, "height": 896, "batch_size": 1 } },
  "13": { "class_type": "KSampler",
          "inputs": { "model": ["1", 0], "positive": ["11", 0], "negative": ["9", 0],
                      "latent_image": ["12", 0], "seed": 42, "steps": 20,
                      "cfg": 1.0, "sampler_name": "euler",
                      "scheduler": "simple", "denoise": 1.0 } },
  "14": { "class_type": "VAEDecode",
          "inputs": { "samples": ["13", 0], "vae": ["3", 0] } },
  "15": { "class_type": "SaveImage",
          "inputs": { "images": ["14", 0], "filename_prefix": "scene" } }
}
```

Важливі моменти:

- Для FLUX `cfg` у KSampler = **1.0**; сила підказки керується окремим вузлом `FluxGuidance` (3.5). Підвищення `cfg` погіршує результат.
- Негативний промпт підключено до вузла `9` як заглушку (FLUX при `cfg=1.0` його ігнорує).
- Використано звичайний `VAEDecode`. Якщо на більших роздільностях виникне OOM — перейдіть на `VAEDecodeTiled` з параметрами `tile_size`, `overlap`, `temporal_size`, `temporal_overlap`.
- Назви вузлів GGUF (`UnetLoaderGGUF`, `DualCLIPLoaderGGUF`) залежать від версії розширення. Перевірити фактичні назви:

```bash
curl -s http://127.0.0.1:8188/object_info | grep -io '"UnetLoaderGGUF"\|"DualCLIPLoaderGGUF"'
```

### Роздільність

- Базова: **896×896** (безпечно для 12 ГБ).
- FLUX навчений на ~1 МП. Для 16:9 краще генерувати близько 1 МП (напр. **1344×768**) і за потреби апскейлити окремим проходом, ніж одразу 1920×1080 (можливі артефакти композиції та OOM).

## 5. Файл сцен (scenes.txt)

Один промпт на рядок; рядки, що починаються з `#`, пропускаються. Акцент на дії та оточенні; чоловік — фігура середнього плану, а не позуюча модель. Приклад рядка:

```
a man herding sheep across an alpine meadow high in the Ukrainian Carpathians; wildflower grass, scattered flock, distant snow-capped peaks dominating the horizon; the man as a mid-ground figure about a third of the frame height, body and action clearly visible but not dominant; hazy warm summer afternoon; wide 16:9 composition, deep depth of field, environment and action as joint focal points, photorealistic, highly detailed
```

Структура промпту: `дія + локація; деталі оточення; масштаб фігури; освітлення; теги якості`.

## 6. Пакетний скрипт (comfy_batch.py)

Скрипт послідовно бере рядки зі `scenes.txt`, підставляє текст у вузол `9` та seed у вузол `13`, відправляє граф на `/prompt`, очікує завершення через `/history`, потім робить паузу для охолодження GPU.

Ключові параметри (на початку файлу):

| Параметр | Значення | Призначення |
|---|---|---|
| `SERVER` | `http://127.0.0.1:8188` | адреса API ComfyUI |
| `WORKFLOW` | `workflow_api.json` | граф у API-форматі |
| `SCENES_FILE` | `scenes.txt` | список промптів |
| `POSITIVE_NODE_ID` | `"9"` | вузол позитивного промпту |
| `KSAMPLER_NODE_ID` | `"13"` | вузол KSampler (seed) |
| `DELAY_SECONDS` | `90` | пауза між завданнями (охолодження) |
| `RANDOM_SEED` | `True` | новий seed на кожну сцену |
| `JOB_TIMEOUT` | `900` | таймаут на одне зображення (сек) |

### Розташування файлів клієнта

Клієнтські файли (`workflow_api.json`, `comfy_batch.py`, `scenes.txt`) тримаємо в окремій теці, **не** в теці ComfyUI:

```
~/comfy_ui_scripts/
├── workflow_api.json
├── comfy_batch.py
├── scenes.txt
└── batch.log
```

Скрипт читає `workflow_api.json` і `scenes.txt` із поточної теки, тож запускати треба саме звідти. Згенеровані зображення зберігає сам сервер ComfyUI у свою `output/` — це не залежить від того, звідки запущено клієнт.

```bash
mkdir -p ~/comfy_ui_scripts
mv workflow_api.json comfy_batch.py scenes.txt ~/comfy_ui_scripts/
```

Запуск (стійкий до розриву SSH-сесії):

```bash
cd ~/comfy_ui_scripts
nohup python comfy_batch.py > batch.log 2>&1 &
tail -f batch.log
```

> `nohup` переживає розрив SSH, але **не** перезавантаження сервера чи OOM-kill. Для повної стійкості запускайте у `tmux`/`screen` або як окремий systemd-юніт.

## 7. Результати

Зображення зберігаються у стандартній теці виводу ComfyUI:

```bash
ls /home/automation/ComfyUI/output/   # scene_00001_.png, scene_00002_.png, ...
```

## Орієнтовний час

При ~25 с на зображення + 90 с паузи одна сцена займає ≈2 хв; 1000 сцен — приблизно **30+ годин** сумарного часу. Плануйте дві нічні сесії або зменшіть `DELAY_SECONDS`.

## Підсумковий перелік файлів

```
~/comfy_ui_scripts/          # клієнтські файли (окрема тека)
├── workflow_api.json        # граф воркфлоу (API-формат)
├── comfy_batch.py           # пакетний драйвер
├── scenes.txt               # 1000 промптів (1 на рядок)
└── batch.log                # лог виконання (nohup)

~/ComfyUI/                   # сам сервер
└── output/                  # згенеровані зображення (scene_*.png)
```

## Майбутні покращення (заплановано)

- Підвищення роздільності до Full HD через додатковий прохід апскейлу (генерація 1344×768 → апскейл до 1920×1080).
- Перехід на більшу модель (Q5_K_M ≈ 8 ГБ) за наявності запасу VRAM.
- За потреби фіксації конкретного обличчя — повернення PuLID-Flux + фото обличчя.
