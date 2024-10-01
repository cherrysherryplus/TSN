# TSN (Target signature network for small object tracking)
This paper has proposed a tracker for single object tracking from view of UAVs. Excellent performance has been validated across 5 commonly adopted benchmarks, i.e _UAV123_10fps_, _UAV20L_, _DTB70_, _VisDrone2020_ and _LaTOT_test_. The training datasets contain _ImageNet DET, ImageNet VID, COCO, LaSOT, GOT-10K_.

Three main contributions are attributed for TSN's outstanding tracking performance.
- Size-Aware Module (**SAM**): Leverage features extensively through multi-receptive learning to adopt size variations of tracked object.
- Center Attention Module (**CAM**): Suppress background distractions and highlight object region for consistent tracking.
- Dynamic Positive sample Definition Strategy (**DPDS**): Adjust positive sampling area subject to scale variation in the training stage for better convergence.

The implementation is based on PySOT and SiamCAR. Questions about environment creation, testing and et. al, can be firstly referred to their official repositories and blogs, which we have found them helpful enough. Corresponding result files of TSN on different benchmarks have been uploaded. Codes of network definition and benchmark evaluation, together with related checkpoint and hyper-params, will be available soon. 

| Result text files | Others |
| --- | --- |
| [百度网盘](https://pan.baidu.com/s/1UxK5qs9LSUMzgVcm3cSBhQ?pwd=chgr) | ~ |

Some visualization of TSN along with Ground Truth annotations are provided. Since UAV20L and VisDrone2019 are too long to be visulaized due to their long-term tracking scenes, we only display tracking results of relatively-short videos from the other 3 datasets.

#### Results on _Animal2_ and _SpeedCar2_ in **DTB70** (Bounding boxes in blue are predicted ones)
https://github.com/user-attachments/assets/85a1fafa-f1e8-4460-b56f-89cc96d3cf37

https://github.com/user-attachments/assets/0989d45d-ece1-43c6-be3a-70cd8884604e

#### Results on _car16_ in **LaTOT**
https://github.com/user-attachments/assets/d1e8edf1-12cf-4e5a-90d6-3a9a1aa5ebe8

#### Results on _person14_3_ and _wakeboard7_ in **UAV123@10fps**
https://github.com/user-attachments/assets/31788942-cb11-43a7-b061-128f3ca02dd5

https://github.com/user-attachments/assets/c0752d31-98ae-4800-9f91-5fc01b4f20a7

Since related paper has been submitted for peer review, codes will be fulfilled once the paper is accepeted.
