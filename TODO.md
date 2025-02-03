## Setup & Testing

### Prerequisites

- Node.js and npm installed on your machine.

### Installation

   ```bash
   git clone https://github.com/[your-github-username]/oocr.git
   cd oocr
   npm install -D vitepress vue
   npm run docs:dev
   ```
### Live Test Documentation Server:
   ```bash
   pip install sphinx-autobuild
   sphinx-autobuild docs/source build/html
   visit: http://localhost:8000
   ```


TODO:

- create CI/CD pipeline in .github/workflows

- adapt directory structure:
oocr/
├── core/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── exceptions.py      # Custom exceptions
│   └── logging.py         # Logging configuration
│
├── data/
│   ├── __init__.py
│   ├── datasets/          # Dataset implementations
│   │   ├── __init__.py
│   │   ├── base.py       # Abstract base dataset
│   │   ├── iam.py        # IAM dataset implementation
│   │   ├── mnist.py      # MNIST dataset implementation
│   │   └── custom.py     # Custom dataset implementation
│   │
│   ├── processors/       # Data processing implementations
│   │   ├── __init__.py
│   │   ├── base.py      # Abstract processor interface
│   │   ├── trocr.py     # TrOCR processor implementation
│   │   └── donut.py     # Donut processor implementation
│   │
│   └── augmentation/    # Data augmentation
│       ├── __init__.py
│       ├── transforms.py
│       └── policies.py
│
├── models/
│   ├── __init__.py
│   ├── base.py          # Abstract model interface
│   ├── trocr.py         # TrOCR model implementation
│   ├── donut.py         # Donut model implementation
│   └── nougat.py        # Nougat model implementation
│
├── generation/
│   ├── __init__.py
│   ├── generator.py     # Text image generator
│   ├── fonts.py        # Font management
│   └── backgrounds.py  # Background management
│
├── metrics/
│   ├── __init__.py
│   ├── base.py        # Abstract metric interface
│   ├── accuracy.py    # Accuracy metrics
│   ├── distance.py    # Distance-based metrics
│   └── evaluator.py   # Evaluation orchestrator
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   ├── io.py
│   └── text.py
│
└── cli/
    ├── __init__.py
    ├── train.py
    ├── inference.py
    └── generate.py

- poetry dependency installation

- look for configuration management system

- todo: theme creation - https://www.sphinx-doc.org/en/master/development/html_themes/index.html#extension-html-theme

