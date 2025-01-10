# F1 Sentiment Analysis and Prediction Project
This project analyzes sentiment data and does predictions on race results, driver team changes and steward disicions for for the r/formula1 subreddit using machine learning and natural language processing (NLP). The workflow includes ABSA sentiment analysis, Bert, vader, gliner and N-gram models.

## Data Setup:
Place raw data files (.ndjson) in the data/raw directory.

## Installation:
Make sure you have Python 3.12.8 installed and install required libraries using:
- `py -3.12 -m venv .venv && source .venv/Scripts/activate` (create and activate virtual environment)
- `pip install -r requirements.txt`

## Contributing
Contributions are welcome! To contribute:

- Fork the repository.
- Create a new branch (git switch -c feature-branch).
- Commit your changes (git commit -m 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Open a Pull Request.

## License:
This project is licensed under the MIT License. See [LICENSE](https://github.com/oThommy/f1-subreddits-nlp/blob/main/LICENSE) for details.