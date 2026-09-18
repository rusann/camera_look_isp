### LLM-управляемый ISP использующий диффузионные модели

## Установка

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
export HF_HOME=/mnt/data/hf_cache
huggingface-cli login
```

Тестируется и обучается на 48 GB GPU с флагом --gpu48.

## Данные

```bash
python download_data.py --list
python download_data.py --targets models,haldclut
python download_data.py --targets fivek_dng --max-files 200 --max-gb 15
python download_data.py --targets fivek_experts --experts c
python build_dataset.py --fresh --scenes 200 --looks 12 --crops 4 --edits 2 --experts c
```

`build_dataset.py` создает пары и манифест в `$CAMERA_LOOK_DATA` (default
`/mnt/data`). Проверить манифест `--audit <manifest>`.

## Обучение

```bash
M=/mnt/data/manifest.jsonl
python camera_look_isp.py --gpu48 --phase frontend      --manifest $M
python camera_look_isp.py --gpu48 --phase lut_vqvae     --manifest $M
python camera_look_isp.py --gpu48 --phase director_sft  --manifest $M --max_steps 6000
python camera_look_isp.py --gpu48 --phase director_sft  --manifest $M \
    --use_lut --lut_only --resume --max_steps 2000
python camera_look_isp.py --gpu48 --phase cache_style   --manifest $M --use_lut
```

`--out_dir` для экспериментов чтобы не перезаписать существующий чекпоинт; `--resume <path>` переиспользует любой чекпоинт.

## Тестирование

```bash
python camera_look_isp.py --gpu48 --phase probe_guide   --manifest $M --use_lut
python camera_look_isp.py --gpu48 --phase eval_prompts  --manifest $M --use_lut
python bench_distill.py --phase benchmark --manifest $M --use_lut
python bench_distill.py --phase heldout   --manifest $M --use_lut --hold 2
```


## Деплой

```bash
python bench_distill.py --phase bake             --manifest $M --use_lut
python bench_distill.py --phase distill_director --manifest $M --use_lut --max_steps 3000
python bench_distill.py --phase apply --image photo.jpg \
    --prompt "make this look like it was shot on a Fujifilm camera" --out demo.jpg --use_lut
```

## Файлы

| файл | комментарий |
|---|---|
| `camera_look_isp.py` | модель, обучение, метрики |
| `build_dataset.py` | создание датасета |
| `download_data.py` | загрузка данных и моделей |
| `bench_distill.py` | бенчмарки, baking, дистилляция, инференс |
