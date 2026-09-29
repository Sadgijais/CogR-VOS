# Literature

Papers collected for this project, grouped by theme: 57 distinct works. The PDFs are not redistributed in this repository. Each title links to a Google Scholar search so you can find the official version.

## Contents

- [1. Referring video object segmentation](#1-referring-video-object-segmentation)
- [2. Grounding-DINO and video grounding](#2-grounding-dino-and-video-grounding)
- [3. SAM 2 and video object tracking](#3-sam-2-and-video-object-tracking)
- [4. Memory-guided video object segmentation](#4-memory-guided-video-object-segmentation)
- [5. VLM-guided video object segmentation](#5-vlm-guided-video-object-segmentation)
- [6. Event-driven reasoning](#6-event-driven-reasoning)
- [Reference numbers](#reference-numbers)

## 1. Referring video object segmentation

Task definition, benchmarks and the closest prior systems.

| Paper | Venue | Takeaway |
|---|---|---|
| [Unleashing the Temporal-Spatial Reasoning Capacity of GPT for Training-Free Audio and Language Referenced Video Object Segmentation (AL-Ref-SAM 2)](https://scholar.google.com/scholar?q=Unleashing+the+Temporal-Spatial+Reasoning+Capacity+of+GPT+for+Training-Free+Audio+and+Language+Referenced+Video+Object+Segmentation+%28AL-Ref-SAM+2%29) | AAAI 2025 | Training-free R-VOS: GPT-4 picks a pivot frame and pivot box among Grounding-DINO candidates, SAM 2 propagates both ways. 74.2 J&F on Ref-DAVIS17. A duplicate copy was also collected under memory-guided VOS. |
| [Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation](https://scholar.google.com/scholar?q=Long-RVOS%3A+A+Comprehensive+Benchmark+for+Long-term+Referring+Video+Object+Segmentation) | CVPR 2026 | Benchmark built around occlusion, disappearance and reappearance in long videos. |
| [Robust Referring Video Object Segmentation with Cyclic Structural Consensus](https://scholar.google.com/scholar?q=Robust+Referring+Video+Object+Segmentation+with+Cyclic+Structural+Consensus) | ICCV 2023 | Robustness to expressions that refer to nothing in the video, via cycle consistency. |
| [Segment Every Reference Object in Spatial and Temporal Spaces](https://scholar.google.com/scholar?q=Segment+Every+Reference+Object+in+Spatial+and+Temporal+Spaces) | ICCV 2023 | Unified framework for image-level and video-level referring segmentation. |
| [OnlineRefer: A Simple Online Baseline for Referring Video Object Segmentation](https://scholar.google.com/scholar?q=OnlineRefer%3A+A+Simple+Online+Baseline+for+Referring+Video+Object+Segmentation) | ICCV 2023 | Online, frame-by-frame R-VOS baseline that propagates queries through time. |
| [ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](https://scholar.google.com/scholar?q=ReferDINO%3A+Referring+Video+Object+Segmentation+with+Visual+Grounding+Foundations) | ICCV 2025 | Builds R-VOS on a grounding foundation model. |
| [ReferEverything: Towards Segmenting Everything We Can Speak of in Videos](https://scholar.google.com/scholar?q=ReferEverything%3A+Towards+Segmenting+Everything+We+Can+Speak+of+in+Videos) | ICCV 2025 | Open-ended referring segmentation across a broad range of concepts in video. |
| [InterRVOS: Interaction-Aware Referring Video Object Segmentation](https://scholar.google.com/scholar?q=InterRVOS%3A+Interaction-Aware+Referring+Video+Object+Segmentation) | CVPR 2026 | R-VOS that models interactions between objects named in the expression. |
| [SLVP: Self-Supervised Language-Video Pre-Training for Referring Video Object Segmentation](https://scholar.google.com/scholar?q=SLVP%3A+Self-Supervised+Language-Video+Pre-Training+for+Referring+Video+Object+Segmentation) | WACVW 2024 | Self-supervised language-video pre-training to improve R-VOS. |
| [Fine-grained Spatiotemporal Grounding on Egocentric Videos](https://scholar.google.com/scholar?q=Fine-grained+Spatiotemporal+Grounding+on+Egocentric+Videos) | ICCV 2025 | Spatio-temporal grounding of fine-grained language in egocentric video. |
| [TSAM: Temporal SAM Augmented with Multimodal Prompts for Referring Audio-Visual Segmentation](https://scholar.google.com/scholar?q=TSAM%3A+Temporal+SAM+Augmented+with+Multimodal+Prompts+for+Referring+Audio-Visual+Segmentation) | CVPR 2025 | Adapts SAM to referring audio-visual segmentation with text and audio prompts. A duplicate copy was also collected under SAM 2 tracking. |

## 2. Grounding-DINO and video grounding

The open-vocabulary detector that turns a sentence into candidate boxes.

| Paper | Venue | Takeaway |
|---|---|---|
| [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://scholar.google.com/scholar?q=Grounding+DINO%3A+Marrying+DINO+with+Grounded+Pre-Training+for+Open-Set+Object+Detection) | ECCV 2024 | Open-set detector that takes free text and returns matching boxes with scores. |
| [Grounding DINO 1.5: Advance the Edge of Open-Set Object Detection](https://scholar.google.com/scholar?q=Grounding+DINO+1.5%3A+Advance+the+Edge+of+Open-Set+Object+Detection) | arXiv 2024 | Stronger and edge-deployable variants of Grounding DINO. |
| [VideoGrounding-DINO: Towards Open-Vocabulary Spatio-Temporal Video Grounding](https://scholar.google.com/scholar?q=VideoGrounding-DINO%3A+Towards+Open-Vocabulary+Spatio-Temporal+Video+Grounding) | CVPR 2024 | Extends Grounding DINO to spatio-temporal grounding in video. |
| [Unlocking the Potential of Grounding DINO in Videos: Parameter-Efficient Adaptation](https://scholar.google.com/scholar?q=Unlocking+the+Potential+of+Grounding+DINO+in+Videos%3A+Parameter-Efficient+Adaptation) | CVPRW 2026 | Parameter-efficient adaptation of Grounding DINO for video. |
| [Think Before You Segment: An Object-aware Reasoning Agent for Referring Audio-Visual Segmentation](https://scholar.google.com/scholar?q=Think+Before+You+Segment%3A+An+Object-aware+Reasoning+Agent+for+Referring+Audio-Visual+Segmentation) | arXiv 2025 | Agent that reasons about candidate objects before segmenting. |
| [Low-Rank Prompt Adaptation for Open-Vocabulary Object Detection](https://scholar.google.com/scholar?q=Low-Rank+Prompt+Adaptation+for+Open-Vocabulary+Object+Detection) | ICCVW 2025 | Low-rank prompt adaptation for open-vocabulary detectors. Two copies were collected, differing only in page numbers. |
| [Few-Shot Adaptation of Grounding DINO for Agricultural Domain](https://scholar.google.com/scholar?q=Few-Shot+Adaptation+of+Grounding+DINO+for+Agricultural+Domain) | CVPRW 2025 | Few-shot domain adaptation of Grounding DINO. |

## 3. SAM 2 and video object tracking

SAM 2-based trackers, distractor handling and efficiency.

| Paper | Venue | Takeaway |
|---|---|---|
| [SAMURAI: Motion-Aware Memory for Training-Free Visual Object Tracking with SAM 2](https://scholar.google.com/scholar?q=SAMURAI%3A+Motion-Aware+Memory+for+Training-Free+Visual+Object+Tracking+with+SAM+2) | IEEE TIP 2026 | Training-free SAM 2 tracker: Kalman motion scores for mask selection and score-gated memory writes. |
| [A Distractor-Aware Memory for Visual Object Tracking with SAM2 (DAM4SAM)](https://scholar.google.com/scholar?q=A+Distractor-Aware+Memory+for+Visual+Object+Tracking+with+SAM2+%28DAM4SAM%29) | CVPR 2025 | Splits memory into recent-appearance and distractor-resolving parts; introduces the DiDi dataset. |
| [The Third Visual Object Tracking and Segmentation Challenge Results (VOTS2025)](https://scholar.google.com/scholar?q=The+Third+Visual+Object+Tracking+and+Segmentation+Challenge+Results+%28VOTS2025%29) | ICCVW 2025 | State of the field: 80% of entries are SAM 2-based; defines NRE, DRE and ADQ; winner failures split about half drift, half false absence. |
| [SAM 2++: Tracking Anything at Any Granularity](https://scholar.google.com/scholar?q=SAM+2%2B%2B%3A+Tracking+Anything+at+Any+Granularity) | arXiv 2025 | One model for mask, box and point tracking; authors note remaining failures under occlusion and similar distractors. |
| [Enhanced Kalman with Adaptive Appearance Motion SORT for Grounded Generic Multiple Object Tracking (KAM-SORT)](https://scholar.google.com/scholar?q=Enhanced+Kalman+with+Adaptive+Appearance+Motion+SORT+for+Grounded+Generic+Multiple+Object+Tracking+%28KAM-SORT%29) | ACCV 2024 | Language-described tracking with appearance and motion weights adapted to how alike the detections are. |
| [AITrack: Attention-Based Image-Text Alignment for Visual Tracking](https://scholar.google.com/scholar?q=AITrack%3A+Attention-Based+Image-Text+Alignment+for+Visual+Tracking) | IEEE Access 2025 | Vision-language tracker using an ROI text-guided encoder and simple linear alignment. |
| [EntitySAM: Segment Everything in Video](https://scholar.google.com/scholar?q=EntitySAM%3A+Segment+Everything+in+Video) | CVPR 2026 | Prompt-free video segmentation on frozen SAM 2 with an added DINOv2 encoder. |
| [Q-MiniSAM2: A Quantization-based Benchmark for Resource-Efficient Video Segmentation](https://scholar.google.com/scholar?q=Q-MiniSAM2%3A+A+Quantization-based+Benchmark+for+Resource-Efficient+Video+Segmentation) | IJCAI 2025 | Post-training quantization of SAM 2; keeps 93% of full-precision quality at 6 bits. |
| [AuralSAM2: Enabling SAM2 Hear Through Pyramid Audio-Visual Feature Prompting](https://scholar.google.com/scholar?q=AuralSAM2%3A+Enabling+SAM2+Hear+Through+Pyramid+Audio-Visual+Feature+Prompting) | CVPR 2026 Findings | Audio prompting of a frozen SAM 2; identifies audio prompt dilution across layers. |

## 4. Memory-guided video object segmentation

How memory banks are built, bounded and updated.

| Paper | Venue | Takeaway |
|---|---|---|
| [Putting the Object Back into Video Object Segmentation (Cutie)](https://scholar.google.com/scholar?q=Putting+the+Object+Back+into+Video+Object+Segmentation+%28Cutie%29) | CVPR 2024 | Object-level memory reading; the object memory is not updated while the target is invisible. |
| [RMem: Restricted Memory Banks Improve Video Object Segmentation](https://scholar.google.com/scholar?q=RMem%3A+Restricted+Memory+Banks+Improve+Video+Object+Segmentation) | CVPR 2024 | Showed that growing memory dilutes attention; a small, curated bank works better. |
| [Domain Generalization for Multiple Video Object Segmentation and Tracking Using Transformers and Smart Memory (MuSMem)](https://scholar.google.com/scholar?q=Domain+Generalization+for+Multiple+Video+Object+Segmentation+and+Tracking+Using+Transformers+and+Smart+Memory+%28MuSMem%29) | IJCV 2026 | Information-preserving memory deletion with O(1) memory and a DINOv2 accept/reject check. |
| [LiVOS: Light Video Object Segmentation with Gated Linear Matching](https://scholar.google.com/scholar?q=LiVOS%3A+Light+Video+Object+Segmentation+with+Gated+Linear+Matching) | CVPR 2025 | Constant-size gated linear-attention memory; 53% less GPU memory. |
| [Efficient Video Object Segmentation via Modulated Cross-Attention Memory (MAVOS)](https://scholar.google.com/scholar?q=Efficient+Video+Object+Segmentation+via+Modulated+Cross-Attention+Memory+%28MAVOS%29) | WACV 2025 | Two-slot modulated memory; 87% less GPU memory than DeAOT-L on long videos. |
| [Alignment Before Aggregation: Trajectory Memory Retrieval Network for Video Object Segmentation](https://scholar.google.com/scholar?q=Alignment+Before+Aggregation%3A+Trajectory+Memory+Retrieval+Network+for+Video+Object+Segmentation) | ICCV 2023 | Aligns memory frames spatially, then aggregates along time; agent-level correlation suppresses look-alike matches. |
| [DeVOS: Flow-Guided Deformable Transformer for Video Object Segmentation](https://scholar.google.com/scholar?q=DeVOS%3A+Flow-Guided+Deformable+Transformer+for+Video+Object+Segmentation) | WACV 2024 | Matching with flow-guided deformable attention that separates motion from semantics. |
| [Unsupervised Video Object Segmentation via Prototype Memory Network](https://scholar.google.com/scholar?q=Unsupervised+Video+Object+Segmentation+via+Prototype+Memory+Network) | WACV 2023 | Part-level prototype memory with a learned usefulness score. |
| [Multi-grained Temporal Prototype Learning for Few-shot Video Object Segmentation](https://scholar.google.com/scholar?q=Multi-grained+Temporal+Prototype+Learning+for+Few-shot+Video+Object+Segmentation) | ICCV 2023 | Clip, frame and memory prototypes with reliability-based memory selection. |
| [Towards Open-Vocabulary Video Instance Segmentation (OV2Seg)](https://scholar.google.com/scholar?q=Towards+Open-Vocabulary+Video+Instance+Segmentation+%28OV2Seg%29) | ICCV 2023 | Introduces LV-VIS; memory queries updated with an object-score gate so occluded objects are not corrupted. |
| [LVOS: A Benchmark for Long-term Video Object Segmentation (supplementary)](https://scholar.google.com/scholar?q=LVOS%3A+A+Benchmark+for+Long-term+Video+Object+Segmentation+%28supplementary%29) | ICCV 2023 | Ablation of reference, global and local memory banks; long-term reappearance is the hardest attribute. |

## 5. VLM-guided video object segmentation

LLM and VLM reasoning for R-VOS, open-vocabulary segmentation, and efficiency.

| Paper | Venue | Takeaway |
|---|---|---|
| [VISA: Reasoning Video Object Segmentation via Large Language Models](https://scholar.google.com/scholar?q=VISA%3A+Reasoning+Video+Object+Segmentation+via+Large+Language+Models) | ECCV 2024 | Defines ReasonVOS and the ReVOS benchmark (including nonexistent-object queries); one pivot frame, then propagate. |
| [ViLLa: Video Reasoning Segmentation with Large Language Model](https://scholar.google.com/scholar?q=ViLLa%3A+Video+Reasoning+Segmentation+with+Large+Language+Model) | ICCV 2025 | Key-segment extraction and multi-level segmentation tokens; 74.3 J&F on Ref-DAVIS17. |
| [InstructSeg: Unifying Instructed Visual Segmentation with Multi-Modal Large Language Models](https://scholar.google.com/scholar?q=InstructSeg%3A+Unifying+Instructed+Visual+Segmentation+with+Multi-Modal+Large+Language+Models) | ICCV 2025 | Compact end-to-end 3B model covering image and video referring and reasoning segmentation; supplement also collected. |
| [Segment Countless Visual Concepts without Training Endeavor (CLIP as RNN)](https://scholar.google.com/scholar?q=Segment+Countless+Visual+Concepts+without+Training+Endeavor+%28CLIP+as+RNN%29) | CVPR 2024 | Training-free recurrent loop over frozen CLIP that prunes low-confidence queries until stable. |
| [OpenVIS: Open-vocabulary Video Instance Segmentation (InstFormer)](https://scholar.google.com/scholar?q=OpenVIS%3A+Open-vocabulary+Video+Instance+Segmentation+%28InstFormer%29) | AAAI 2025 | Per-instance CLIP embeddings in one pass and a tracker that predicts the next identity token. |
| [Unified Embedding Alignment for Open-Vocabulary Video Instance Segmentation (OVFormer)](https://scholar.google.com/scholar?q=Unified+Embedding+Alignment+for+Open-Vocabulary+Video+Instance+Segmentation+%28OVFormer%29) | ECCV 2024 | Aligns instance queries with CLIP embeddings and trains at video level. |
| [Towards Real-Time Open-Vocabulary Video Instance Segmentation (TROY-VIS)](https://scholar.google.com/scholar?q=Towards+Real-Time+Open-Vocabulary+Video+Instance+Segmentation+%28TROY-VIS%29) | WACV 2025 | Cached text embeddings and key-frame kernel interpolation give a 20x speedup. |
| [CaptionFormer: Unified Segmentation, Tracking and Captioning for Spatio-Temporal Objects](https://scholar.google.com/scholar?q=CaptionFormer%3A+Unified+Segmentation%2C+Tracking+and+Captioning+for+Spatio-Temporal+Objects) | CVPR 2026 | Dense video object captioning with score-weighted temporal aggregation; a zero-shot frontier VLM scores only 6.6 CHOTA. |
| [ViCaS: A Dataset for Combining Holistic and Pixel-level Video Understanding](https://scholar.google.com/scholar?q=ViCaS%3A+A+Dataset+for+Combining+Holistic+and+Pixel-level+Video+Understanding) | CVPR 2025 | Human-written captions grounded to pixel-accurate masks; LLM judges track human ratings better than BLEU or CIDEr. |

## 6. Event-driven reasoning

Detecting events online, reasoning about them, and invoking expensive computation selectively.

| Paper | Venue | Takeaway |
|---|---|---|
| [Event-VStream: Event-Driven Real-Time Understanding for Long Video Streams](https://scholar.google.com/scholar?q=Event-VStream%3A+Event-Driven+Real-Time+Understanding+for+Long+Video+Streams) | CVPR 2026 Findings | Decodes only at motion and semantic event boundaries with an adaptive threshold; +10.4 points on a frozen backbone. |
| [Think Before You Simulate: Symbolic Reasoning to Orchestrate Neural Computation for Counterfactual Question Answering (CRCG)](https://scholar.google.com/scholar?q=Think+Before+You+Simulate%3A+Symbolic+Reasoning+to+Orchestrate+Neural+Computation+for+Counterfactual+Question+Answering+%28CRCG%29) | WACV 2024 | Runs the expensive simulator only from the first causally affected frame. |
| [Knowing Where to Focus: Event-aware Transformer for Video Grounding (EaTR)](https://scholar.google.com/scholar?q=Knowing+Where+to+Focus%3A+Event-aware+Transformer+for+Video+Grounding+%28EaTR%29) | ICCV 2023 | Video-specific event queries from slot attention; self-similarity-based event boundaries. |
| [MECD+: Unlocking Event-Level Causal Graph Discovery for Video Reasoning](https://scholar.google.com/scholar?q=MECD%2B%3A+Unlocking+Event-Level+Causal+Graph+Discovery+for+Video+Reasoning) | TPAMI 2026 | Granger-style mask-and-compare test for event causality, with fixes for confounding and illusory causality. |
| [Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering (CMCIR)](https://scholar.google.com/scholar?q=Cross-Modal+Causal+Relational+Reasoning+for+Event-Level+Visual+Question+Answering+%28CMCIR%29) | TPAMI 2023 | Front-door and back-door interventions to remove visual and linguistic confounding in VideoQA. |
| [Dynamic Spatio-Temporal Graph Reasoning for VideoQA with Self-Supervised Event Recognition](https://scholar.google.com/scholar?q=Dynamic+Spatio-Temporal+Graph+Reasoning+for+VideoQA+with+Self-Supervised+Event+Recognition) | 2024 | Question-guided spatial and temporal graphs with a self-supervised event-recognition task. |
| [Momentor: Advancing Video Large Language Model with Fine-Grained Temporal Reasoning](https://scholar.google.com/scholar?q=Momentor%3A+Advancing+Video+Large+Language+Model+with+Fine-Grained+Temporal+Reasoning) | ICML 2024 | Continuous temporal token space and the Moment-10M instruction dataset. |
| [MotionMAE: Self-supervised Video Representation Learning with Motion-Aware Masked Autoencoders](https://scholar.google.com/scholar?q=MotionMAE%3A+Self-supervised+Video+Representation+Learning+with+Motion-Aware+Masked+Autoencoders) | BMVC 2024 | Adds a temporal-difference reconstruction head; over 3% gain on DAVIS 2017. |
| [ExACT: Language-guided Conceptual Reasoning and Uncertainty Estimation for Event-based Action Recognition](https://scholar.google.com/scholar?q=ExACT%3A+Language-guided+Conceptual+Reasoning+and+Uncertainty+Estimation+for+Event-based+Action+Recognition) | CVPR 2024 | Event-camera (DVS) action recognition with adaptive event representation and distributional uncertainty. |
| [EventGPT: Event Stream Understanding with Multimodal Large Language Models](https://scholar.google.com/scholar?q=EventGPT%3A+Event+Stream+Understanding+with+Multimodal+Large+Language+Models) | CVPR 2025 | First MLLM for event-camera (DVS) streams, trained with a three-stage curriculum. |

## Reference numbers

J&F on Ref-DAVIS17 / Ref-YouTube-VOS / MeViS / ReVOS, as reported in the papers above.

| System | Ref-DAVIS17 | Ref-YouTube-VOS | MeViS | ReVOS | Regime |
|---|---|---|---|---|---|
| CaR (CLIP as RNN) | 30.3 | - | - | - | Training-free |
| Grounded-SAM 2 | 66.2 | 64.8 | 38.9 | - | Training-free |
| AL-Ref-SAM 2 | 74.2 | 67.9 | 42.8 | - | Training-free |
| VISA-13B | 70.4 | 63.0 | 44.5 | 47.5 | MLLM-trained |
| InstructSeg-3B | 71.1 | 67.5 | - | 54.5 | MLLM-trained |
| ViLLa | 74.3 | 67.5 | 49.4 | 57.0 | MLLM-trained |
| DsHmp | 64.9 | 67.1 | 46.4 | - | Specialist |

Note: "event" is used in three senses in this literature: a semantic state transition (used in this project), an event-camera (DVS) stream, and a physical collision in a simulated scene.
