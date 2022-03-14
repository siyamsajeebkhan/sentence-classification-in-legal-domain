After decompressing just run the following commands inside the code folder in the terminal:

```
python3 -m venv test #Used Python 3.6.9 originally
source test/bin/activate
pip3 install -U pip setuptools
pip3 install -r requirements.txt
python3 -m spacy download en
python3 analyze.py <path to txt file>
```

Six notebook files are also provided in the notebooks directory to showcase the effort put in
analysis. To run the notebooks please follow steps 1-3, then run `pip3 install -r
requirements-notebook.txt` and finally run step 5 which should allow you to execute
the notebooks successfully.
