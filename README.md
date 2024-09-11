# TSN (Target signature network for small object tracking)
This paper has proposed a tracker for single object tracking from view of UAVs. Excellent performance has been validated across 5 commonly adopted benchmarks, i.e _UAV123_10fps_, _UAV20L_, _DTB70_, _VisDrone2020_ and _LaTOT_test_. The training datasets contain _ImageNet DET, ImageNet VID, COCO, LaSOT, GOT-10K_.

Three main contributions are attributed for TSN's outstanding tracking performance.
- Size-Aware Module (**SAM**): Leverage features extensively through multi-receptive learning to adopt size variations of tracked object.
- Center Attention Module (**CAM**): Suppress background distractions and highlight object region for consistent tracking.
- Dynamic Positive sample Definition Strategy (**DPDS**): Adjust positive sampling area subject to scale variation in the training stage for better convergence.

The implementation is based on PySOT and SiamCAR. Questions about environment creation, testing and et. al, can be firstly referred to their official repositories and blogs, which we have found them helpful enough. Corresponding checkpoints and result text files of TSN on different benchmarks will be uploaded. Codes of network definition and benchmark evaluation will be available soon. 

Since related paper has been submitted for peer review, codes will be fulfilled once the paper is accepeted.
