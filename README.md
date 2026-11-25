# Подсчёт посетителей и оценка очередей

Учебная НИР. В `01.docx` постановка задачи, в `02.ipynb` анализ данных, в `03.ipynb` базовое решение и проверка. `baseline.py` обрабатывает видео.

## Запуск в WSL

Положите своё видео в корень проекта под именем `video.mp4`. Выполните из этой же папки:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python baseline.py video.mp4 --output queue.jsonl
```

Результат записывается в `queue.jsonl`. Зоны задаются параметрами `--waiting x1 y1 x2 y2` и `--service x1 y1 x2 y2` в пикселях. Если известна площадь зоны ожидания, передайте её в м² через `--area`.

Для ноутбуков распакуйте [MOT17](https://motchallenge.net/data/MOT17/) и [MOT20](https://motchallenge.net/data/MOT20/) в `data/`, чтобы были папки `data/MOT17/train` и `data/MOT20/train`. Достаточно файлов `seqinfo.ini`, `gt/gt.txt` и `det/det.txt`. Из [архива ShanghaiTech](https://www.kaggle.com/datasets/tthien/shanghaitech) поместите только папки `part_A` и `part_B` в `data/ShanghaiTech`. Запустите ноутбуки в VS Code с ядром `.venv/bin/python`.
