# BBO-Lin-Parser
Project to parse BBO lin files producing a table of the bidding sequences they contain. BBO is an online contract duplicate bridge gaming website. This project is not supported. Not end user friendly. Requires python programming knowledge.

The purpose of the project is create a baseline of BBO's bidding sequences. The baseline can serve as a guide for other bidding systems or be used for statistical analysis.

The project consists of a jupyter notebook (python) file. The notebook reads a glob of .lin files, wrangles their bidding announcements, creating bbo_bidding_sequences_table.py which is a table of bidding sequences. About 1M bidding sequences are produced.

# Dependencies:
- Python 3.8+
- At least 64GB of memory.

# To install:
    pip install -r requirements.txt

# To run:
    jupyter notebook bbo_parse_lin_files.ipynb

## Related Projects
For a list of related projects see: https://github.com/BSalita/BridgeStats
