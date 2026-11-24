"""Подсчёт видимых пешеходов и оценка очереди."""
import argparse
import json

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self):
        self.active = {}
        self.next_id = 1

    def update(self, frame, boxes):
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.active = {key: track for key, track in self.active.items()
                       if frame - track[0] <= 4}
        ids = list(self.active)
        assigned = {}
        if ids and len(boxes):
            old = np.array([self.active[key][1] for key in ids])
            start = np.maximum(old[:, None, :2], boxes[None, :, :2])
            end = np.minimum((old[:, :2] + old[:, 2:])[:, None],
                             (boxes[:, :2] + boxes[:, 2:])[None, :])
            intersection = np.maximum(end - start, 0).prod(axis=2)
            union = old[:, 2:].prod(axis=1)[:, None] + boxes[:, 2:].prod(axis=1)
            iou = intersection / (union - intersection)
            rows, cols = linear_sum_assignment(np.where(iou >= .3, iou, 0), maximize=True)
            assigned = {col: ids[row] for row, col in zip(rows, cols) if iou[row, col] >= .3}

        tracks = []
        for index, box in enumerate(boxes):
            key = assigned.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
            self.active[key] = frame, box
            tracks.append((key, box))
        return tracks


class Queue:
    def __init__(self, waiting, service, fps, area=None):
        self.waiting = waiting
        self.service = service
        self.fps = fps
        self.area = area
        self.entered = {}

    def update(self, frame, tracks, active):
        self.entered = {key: start for key, start in self.entered.items() if key in active}
        waiting = 0
        finished = []
        for key, (x, y, w, h) in tracks:
            foot_x, foot_y = x + w / 2, y + h
            x1, y1, x2, y2 = self.service
            start = self.entered.get(key)
            if x1 <= foot_x < x2 and y1 <= foot_y < y2:
                if start is not None:
                    finished.append((frame - self.entered.pop(key)) / self.fps)
                continue
            x1, y1, x2, y2 = self.waiting
            if x1 <= foot_x < x2 and y1 <= foot_y < y2:
                waiting += 1
                self.entered.setdefault(key, frame)
            else:
                self.entered.pop(key, None)
        return {
            "visible": len(tracks),
            "queue_length": waiting,
            "density_m2": waiting / self.area if self.area else None,
            "completed_wait_s": finished,
        }


def analyze(video, waiting, service, area, output):
    tracker = Tracker()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    capture = cv2.VideoCapture(video)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Не удалось открыть видео или определить частоту кадров")
        queue = Queue(waiting, service, fps, area)
        frame = 0
        with open(output, "w", encoding="utf-8") as result:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                frame += 1
                height, width = image.shape[:2]
                window_width, window_height = hog.winSize
                boxes = []
                if width >= window_width and height >= window_height:
                    boxes, _ = hog.detectMultiScale(image, winStride=(8, 8), scale=1.05)
                tracks = tracker.update(frame, boxes)
                row = queue.update(frame, tracks, tracker.active)
                print(json.dumps({"frame": frame, **row}), file=result)
        if frame == 0:
            raise ValueError("В видео нет кадров")
        return frame
    finally:
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--waiting", type=int, nargs=4, default=[10, 80, 300, 360])
    parser.add_argument("--service", type=int, nargs=4, default=[300, 80, 400, 360])
    parser.add_argument("--area", type=float)
    parser.add_argument("--output", default="queue.jsonl")
    args = parser.parse_args()
    for x1, y1, x2, y2 in (args.waiting, args.service):
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            parser.error("Зона задаётся координатами x1 y1 x2 y2 с положительными шириной и высотой")
    if args.area is not None and (not np.isfinite(args.area) or args.area <= 0):
        parser.error("Площадь зоны должна быть положительной")
    try:
        print(f"Обработано кадров: {analyze(args.video, args.waiting, args.service, args.area, args.output)}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
