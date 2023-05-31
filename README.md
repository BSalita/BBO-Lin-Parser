# BBO-Lin-Parser
Project to read and parse a glob of BBO lin files to produce a table of their embedded bidding sequences. BBO is an online contract duplicate bridge gaming website. This project is unsupported and not end user friendly. Requires python programming knowledge.

The purpose of the project is create a baseline of BBO's bidding sequences. The baseline can serve as a guide for creating a baseline of other bidding systems, for statistical analysis, or creation of bridge bidding robots.

The project's author believes this approach, given the additional work of curation of data, will result in super-human bridge bidding abilities. It is unclear whether this approach is sufficient for super-human abilities or whether a neural network is additionally needed to intuit the optimal bids.

The project consists of a jupyter notebook (python) file. The notebook reads a glob of .lin files, wrangles their bidding announcements, creates bbo_bidding_sequences_table.py which is a table of bidding sequences. About 1M bidding sequences are produced.

# Dependencies:
- Python 3.8+
- At least 64GB of memory.

# To install:
    pip install -r requirements.txt

# To run:
    jupyter notebook bbo_parse_lin_files.ipynb

## Related Projects
For a list of related projects see: https://github.com/BSalita/BridgeStats
