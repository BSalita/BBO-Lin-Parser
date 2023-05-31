# BBO-Lin-Parser
Project to read and parse a glob of BBO lin files to produce a table of their embedded bidding sequences. BBO is an online contract duplicate bridge gaming website. This project is unsupported and not end user friendly. Requires python programming knowledge.

The purpose of the project is create a baseline of BBO's bidding rules. The baseline can serve as a guide for creating a baseline for other bidding systems, for statistical analysis, or creation of bridge bidding robots.

The project's author believes this discrete rules approach, given the additional work of curation of data, will result in super-human bridge bidding abilities. It is unclear whether rules alone are sufficient for super-human abilities or must be supplemented by a neural network to best intuit optimal bids.

The project consists of a jupyter notebook (python) file. The notebook reads a glob of .lin files, wrangles their bidding announcements, creates bbo_bidding_sequences_table.py which is a table of bidding sequences and their rules. About 500K bidding sequences are produced.

BBO .lin files can be downloaded using https://github.com/BSalita/BBO-Downloader.

The curent status is proof-of-concept stage.  Months of work are needed to curate data along with development of coverage and validation software.

This project is not affiliated with or sponsored by BBO.

# Dependencies:
- Python 3.8+
- At least 64GB of memory.

# To install:
    pip install -r requirements.txt

# To run:
Install jupyter notebook using pip, conda or mini-conda. Do a search for the installation instructions.

    jupyter notebook bbo_parse_lin_files.ipynb

## Related Projects
For a list of related projects see: https://github.com/BSalita/BridgeStats
