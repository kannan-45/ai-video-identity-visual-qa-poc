\# Identity \& Visual Consistency QA



AI Video Generation POC module for checking identity and visual consistency between a reference image and generated output.



\## Purpose



This module evaluates whether a generated image remains visually and semantically consistent with a reference image.



The QA engine produces:



\- Identity similarity

\- Structural similarity

\- Perceptual similarity

\- Semantic similarity

\- PASS/FAIL decision

\- Reason codes

\- JSON QA report



\## Components



\### Identity



\- AdaFace identity similarity

\- Face detection

\- Prototype face comparison evidence



\### Visual Similarity



\- SSIM

\- LPIPS

\- CLIP



\## Project Structure



```text

identity\_visual\_qa\_project/

│

├── src/

│   ├── qa\_engine.py

│   ├── adaface\_checker.py

│   ├── face\_identity\_checker.py

│   ├── face\_detector.py

│   ├── ssim\_checker.py

│   ├── lpips\_checker.py

│   ├── clip\_checker.py

│   ├── comparison\_ui.py

│   └── test\_fixtures.py

│

├── mock\_data/

│   ├── inputs/

│   │   └── adaface/

│   ├── expected/

│   ├── failures/

│   ├── media/

│   ├── metadata/

│   │   └── manifest.json

│   ├── generate\_fixtures.py

│   └── manifest.json

│

└── README.md

