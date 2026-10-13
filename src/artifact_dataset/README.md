# Artefact detection dataset

The images and labels that trained and tested
`src/cave_explorer/models/artifact_yolov8n_autolabel.pt`, in the standard YOLO
layout. Built with `tools/autolabel` (see its README for how). `COLCON_IGNORE`
keeps colcon from scanning it.

All images come from the robot's own camera (720 x 480, 120 deg) during full
autonomous runs in the simulator. Boxes come from Gazebo's bounding-box and
semantic segmentation cameras, which share the RGB camera's pose and stamps; a
box is kept only if at least 150 of its pixels show the object. Frames were
thinned by robot pose so near-identical views are not repeated.

| Split | Source | Images | Negatives | white_sphere | blue_mushrooms | green_alien | ice_formation | green_crystals |
|---|---|---|---|---|---|---|---|---|
| train | random layouts 1-3 | 1207 | 111 | 349 | 456 | 282 | 456 | 431 |
| val | random layout 4 | 331 | 37 | 114 | 101 | 68 | 138 | 79 |
| test | original `mars_cave.sdf` layout | 311 | 46 | 64 | 87 | 51 | 107 | 77 |

The random layouts move every artefact to another free place in the cave, so
the model learns what artefacts look like rather than where they are; the test
split is the original layout and was never trained on.

Review (`reviews/`): 200 training frames sampled by stratum (small boxes, low
fill, other) had one error, a far green_alien hidden behind a near one; frames
where a box lies mostly inside a larger box of its class were then left out
everywhere. All 315 test frames were checked and 4 unrecognisable slivers
removed.

## Contents

```
data.yaml                 classes and split folders
images/{train,val,test}   JPEG frames
labels/{train,val,test}   YOLO labels: class cx cy w h (normalised)
meta_{split}.jsonl        per frame: sim time, robot pose (odom), every box with
                          its visible pixels and fill, including ones too small to label
reviews/                  review_train.json, review_test.json from the review pages
```

Classes: 0 white_sphere, 1 blue_mushrooms, 2 green_alien, 3 ice_formation,
4 green_crystals.

## Use

Pass `data.yaml` by absolute path, so ultralytics finds the folders next to it:

```bash
python3 -c "from ultralytics import YOLO; YOLO('src/cave_explorer/models/artifact_yolov8n_autolabel.pt').val(data='$(realpath src/artifact_dataset/data.yaml)', split='test')"
```

That reproduces the test result: mAP50 0.991, mAP50-95 0.874 (311 images, 386
boxes). Training used YOLOv8n from COCO weights, 640 px, batch 16, seed 0,
patience 20 (best epoch 64).
