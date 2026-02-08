# Hackathon Alignment: IESA–NXP DeepTech Hackathon

This project directly addresses the core themes of the IESA–NXP DeepTech Hackathon by leveraging cutting-edge AI for semiconductor manufacturing optimization.

## 1. Semiconductor Manufacturing Challenges
- **Yield Loss**: Defects like bridges and cracks directly impact wafer yield.
- **Microscopic Scale**: Traditional optical inspection struggles at sub-micron scales; SEM is required but is slow.
- **Data Scarcity**: Real defect data is proprietary. Our physics-based synthetic generation demonstrates a scalable path to training robust models without large labeled datasets.

## 2. AI-Driven Inspection
By replacing manual operator review with a CNN, we achieve:
- **Consistency**: Removing human subjectivity.
- **Speed**: Enabling near real-time feedback loops.
- **24/7 Operation**: Uninterrupted inspection capacity.

## 3. DeepTech Innovation
- **Custom Architecture**: We avoid generic, heavy models (ResNet50) in favor of domain-specific, lightweight custom CNNs optimized for texture analysis.
- **Edge Readiness**: The model is designed from the ground up for low-power edge deployment on NXP i.MX platforms.

## 4. Industry Applicability
This solution is not just academic; it simulates real-world defect classes (CMP scratches, Vias, Line Edge Roughness) critical to modern fab yield management systems.
