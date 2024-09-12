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

Since related paper has been submitted for peer review, codes will be fulfilled once the paper is accepeted.
