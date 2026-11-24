# Сегментация птиц

Автор: **Габидулин Андрей**.

UNet с энкодером ResNet18 (ImageNet1K_V1) и собственным декодером. Все дообучение выполнено на выданном `train`; `val` используется для выбора чекпойнта и порога. Декодер исходно инициализирован случайно. Обученные веса распространяются отдельно от Git через GitHub Release.


Результат на val: **mean IoU 0.829020** на 1 406 изображениях. ONNX на той же val: 0.829020. [Веса лучшей модели](https://github.com/andreygabi/birds-segmentation-hw/releases/tag/v1.0), [отчёт](report.md).

## Установка

Проверенное окружение: Python 3.12, PyTorch 2.8.0, torchvision 0.23.0. Команды выполняются из корня репозитория.

```bash
git clone https://github.com/andreygabi/birds-segmentation-hw.git
cd birds-segmentation-hw
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Данные и веса

[Архив задания](https://drive.google.com/file/d/10f1H2T-5W-BiqabHHtlZ4ASs19TZmg8R/view): 8 382 пары train, 1 406 пар val, по 200 категорий. Данные в репозиторий не включены.

```bash
python download_data.py
python download_weights.py --onnx
```

Первая команда скачивает архив, проверяет SHA256 и распаковывает его в `data/`. Если Google Drive ограничил автоматическое скачивание, скачайте архив вручную в `.runtime/data.zip` и повторите команду. Вторая команда скачивает `weights/best.pt` и `weights/best.onnx` по ссылкам из [манифеста](results/weights_manifest.json), проверяя SHA256. `--onnx` можно опустить для обычного PyTorch.

Фактическая конфигурация выбранного обучения, родительского запуска и порог находятся в [results/selected_config.yaml](results/selected_config.yaml). Для работы готовой модели скачивание ImageNet-весов и повторное обучение не требуются.

## Оценка на val без переобучения

```bash
python evaluate.py --checkpoint weights/best.pt --data-root data --device cpu --batch-size 8 --output results/reproduced_val.json
```

IoU вычисляется в исходном разрешении отдельно для каждого изображения, затем усредняется. Две пустые маски дают 1, одна пустая — 0. CSV с результатом для каждого изображения сохраняется рядом с JSON.

В исходном архиве встречаются промежуточные значения масок 51, 102, 153 и 204. В соответствии с условием и уточнением автора работы целевой класс определяется строго как **первый канал == 255**. Эти же правила применяются при обучении и оценке. Для проверки: `python audit_data.py --data-root data`.

## Маски для новых изображений

```bash
python predict.py --checkpoint weights/best.pt --images-root /path/to/images --output-dir predictions --device cpu --batch-size 8
```

Папка `images` может содержать подпапки птиц и файлы JPG, JPEG, PNG. Предсказание не читает `gt`. Например, `images/bird/a.jpeg` превращается в `predictions/bird/a.png`. Выход — одноканальный PNG с исходными шириной и высотой, значениями 0/255. Одноканальные входы преобразуются в RGB. Совпадение нескольких расширений у одного относительного имени считается ошибкой.

Для GPU замените `--device cpu` на `--device cuda` или `--device mps`. `auto` выбирает доступное устройство.

## Ускоренный инференс

```bash
python predict.py --checkpoint weights/best.pt --backend onnx --onnx-path weights/best.onnx --images-root /path/to/images --output-dir predictions_onnx --batch-size 1 --threads 6
python evaluate.py --checkpoint weights/best.pt --backend onnx --onnx-path weights/best.onnx --data-root data --batch-size 8 --output results/reproduced_onnx.json
python benchmark.py --checkpoint weights/best.pt --onnx-path weights/best.onnx --images-root data/val/images --limit 64 --batch-sizes 1 4 8 --repeats 3 --threads 6
```

ONNX Runtime использует CPUExecutionProvider, оптимизацию графа и FP32. На проверенном CPU быстрее всего batch=1: 1.76× относительно PyTorch с таким же batch. Сравнение с eager PyTorch проводится на том же CPU, с одинаковыми нормализованными тензорами и числом потоков: 6 intra-op, 1 inter-op. Для каждого batch size — два прогрева и медиана трёх полных проходов. Подготовка данных и постпроцессинг вне таймера. Скрипт сохраняет исходные времена и проверяет расхождения логитов и бинарных масок. Это измерение прямого прохода, а не скорости чтения и записи PNG.

Повторный экспорт из обученного чекпойнта:

```bash
python export_onnx.py --checkpoint weights/best.pt --output weights/best.onnx
```

Пространственный размер фиксирован конфигурацией; размер батча динамический. Полная оценка ONNX на val и результаты сравнения приведены в [report.md](report.md).

## Повторение обучения

```bash
python train.py --config configs/01_bce_256.yaml --data-root data --runs-dir runs
python train.py --config configs/02_dice_256.yaml --data-root data --runs-dir runs
python train.py --config configs/03_refine_320.yaml --data-root data --runs-dir runs --init-from runs/dice_256/best.pt
python select_model.py --data-root data --runs-dir runs
```

Первые два запуска отличаются функцией потерь при одинаковых остальных настройках. Третий продолжает собственный лучший Dice-чекпойнт: размер 320, меньший learning rate, повороты и ColorJitter. Выбор запуска по val при пороге 0.5; затем подбор порога на val по сетке 0.30–0.70 с шагом 0.05. Ничьи по порогу разрешаются в пользу большего значения. Новые данные не используются для подбора.

`runs/<имя>/` содержит `best.pt`, `last.pt`, конфигурацию, историю и списки относительных имён train/val. Сохранение атомарное после каждой эпохи; `last.pt` включает состояния оптимизатора и scheduler. Возобновление:

```bash
python train.py --config configs/02_dice_256.yaml --data-root data --runs-dir runs --resume runs/dice_256/last.pt
```

В Colab сохраняйте `--runs-dir` на подключённый Drive, например `/content/drive/MyDrive/birds-hw/runs`. Для третьего запуска `--init-from` также должен указывать на Drive. После прерывания повторите команду с `--resume .../last.pt`: завершённые эпохи не повторяются. Случайные зерна задаются в конфигурациях; точное побитовое совпадение на разных устройствах не гарантируется. `cache_data: false` уменьшает расход RAM за счёт повторного чтения файлов.

## Файлы и проверки

- `src/model.py` — собственные DoubleConv, UpBlock и UNet.
- `src/data.py` — пары, чтение, letterbox, синхронные преобразования.
- `src/metrics.py` — функция потерь, восстановление размера, IoU.
- `results/` — фактические конфигурации, история, метрики, графики и замеры.
- [report.md](report.md) — эксперименты и ограничения; [answers.md](answers.md) — контрольные вопросы.

```bash
python -m unittest discover -s tests -v
```
