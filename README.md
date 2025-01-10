F1 Sentiment Analysis and Prediction Project
This project analyzes sentiment data and does predictions on race results, driver team changes and steward disicions for for the r/formula1 subreddit using machine learning and natural language processing (NLP). The workflow includes ABSA sentiment analysis, Bert, vader, gliner and N-gram models.

Key features:
Sentiment analysis using pretrained models (e.g., Hugging Face transformers).
self trained N-gram model

Directory Structure:

├── data
│   ├── raw                 # Raw input data (e.g., subreddit comments)
│   ├── processed           # Preprocessed or intermediate data
├── scripts                 # Python scripts for analysis
├── models                  # Saved models or model weights
├── notebooks               # Jupyter notebooks for development and visualization
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


Data Setup:
Place raw data files (.ndjson) in the data/raw directory.

Prepare the environment:

source .venv/bin/activate  # For Linux/Mac
.\.venv\Scripts\activate   # For Windows

Install Dependencies Make sure you have Python 3.12.8 installed. Install required packages using:
pip install -r requirements.txt

Run the Analysis:
navigate to the notebooks folder and start Jupyter:

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git switch -c feature-branch).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a Pull Request.

License:
This project is licensed under the MIT License. See LICENSE for details.