# Vision AI ASL Method Exploration

This repository is an exploration of multiple computer-vision approaches for American Sign Language recognition. It is organized as a comparative project rather than a single model implementation, with separate workstreams for neuromorphic event data and more conventional deep-learning baselines.

## What This Repo Explores

The ASL work here spans two distinct problem settings:

- `neuromorphic_team/`: event-camera recognition on the ASL-DVS dataset
- `deeplearning_team/Kohl/`: frame-based and multimodal ASL experiments on image and video data

The main idea is to study how different representations and model families behave on ASL recognition tasks:

- convolutional neural networks
- vision transformers
- capsule networks
- spiking neural networks
- Event2Vec-style token models
- multimodal contrastive learning
- synthetic event generation with EventGAN

## Repository Structure

- [`neuromorphic_team/`](neuromorphic_team): organized ASL-DVS pipeline, analysis outputs, notebooks, and reports
- [`deeplearning_team/Kohl/`](deeplearning_team/Kohl): standalone ASL model experiments from the deep-learning track

## Neuromorphic Track

The neuromorphic branch focuses on event-based ASL recognition using the ASL-DVS dataset. Its core comparison is between:

- a Conv-SNN baseline in [`neuromorphic_team/scnn/`](neuromorphic_team/scnn)
- an Event2Vec-style classifier in [`neuromorphic_team/event2vec/`](neuromorphic_team/event2vec)

That branch also compares three event encodings:

- `rate`
- `latency`
- `delta`

Current headline result from the generated analysis:

- best completed Event2Vec run: `95.57%` test accuracy with `latency`
- best completed Conv-SNN run: `79.18%` test accuracy

Useful entry points:

- [`neuromorphic_team/README.md`](neuromorphic_team/README.md)
- [`neuromorphic_team/PROJECT_OVERVIEW.md`](neuromorphic_team/PROJECT_OVERVIEW.md)
- [`neuromorphic_team/Analytics/report.md`](neuromorphic_team/Analytics/report.md)
- [`neuromorphic_team/Analytics/Vision_AI_Project_Academic_Report.pdf`](neuromorphic_team/Analytics/Vision_AI_Project_Academic_Report.pdf)

## Deep-Learning Track

The deep-learning branch in [`deeplearning_team/Kohl/`](deeplearning_team/Kohl) explores several more conventional ASL model families through standalone training scripts, including:

- [`VisionAI_Final_CNN.py`](deeplearning_team/Kohl/VisionAI_Final_CNN.py)
- [`VisionAI_Final_ViT.py`](deeplearning_team/Kohl/VisionAI_Final_ViT.py)
- [`VisionAI_Final_CapsNet.py`](deeplearning_team/Kohl/VisionAI_Final_CapsNet.py)
- [`VisionAI_Final_Autoencoder.py`](deeplearning_team/Kohl/VisionAI_Final_Autoencoder.py)
- [`VisionAI_Final_Multimodal.py`](deeplearning_team/Kohl/VisionAI_Final_Multimodal.py)
- [`VisionAI_Final_SNN.py`](deeplearning_team/Kohl/VisionAI_Final_SNN.py)
- [`VisionAI_Final_Video.py`](deeplearning_team/Kohl/VisionAI_Final_Video.py)

These scripts represent method exploration across image classification, transformer-based modeling, neuromorphic-inspired learning, and multimodal reasoning for ASL tasks.

## Project Framing

This repository is best read as an exploration of methods for Vision AI on ASL tasks:

- one branch asks how event-based representations affect ASL recognition quality and efficiency
- one branch asks how standard deep-learning architectures perform on image-based ASL recognition
- together, they show a broader investigation into model choice, representation choice, and sensing modality

## Notes

- The neuromorphic branch is the most structured and fully analyzed part of the repo.
- The deep-learning scripts are more standalone and may require path/config cleanup before reuse in a new environment.
- After the repository reorganization, the team folders now reflect the two major lines of work directly.
