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

## Explanation of
This project's main goal is to implement the conversion of bid announcements into a form compatible with pandas eval function.

For example, the file named bbo_bidding_sequences_table.py contains 500K+ bidding tuples. e.g. Dealer opens 1N:

    (152,(),('1N',),'notrump opener. Could have 5M. -- 2-5 !C; 2-5 !D; 2-5 !H; 2-5 !S; 15-17 HCP; 18- total points','2 <= SL_C <= 5 & 2 <= SL_D <= 5 & 2 <= SL_H <= 5 & 2 <= SL_S <= 5 & 15 <= HCP <= 17 & Total_Points <= 18'),

bidding sequence tuples consist of 5 parts:

1. Bidding sequence id. e.g. 152
2. A tuple of zero or more previous bids. e.g.  () or ('p', '1N', 'p', '2C, ...)
3. A tuple of length one which is the candidate bid, e.g. ('1N',)
4. BBO's textual description of the bid, aka announcement, aka bidding explanation. e.g. 'notrump opener. Could have 5M. -- 2-5 !C; 2-5 !D; 2-5 !H; 2-5 !S; 15-17 HCP; 18- total points'
5. The machine usable form of the announcement compatible with pandas eval() function. e.g. '2 <= SL_C <= 5 & 2 <= SL_D <= 5 & 2 <= SL_H <= 5 & 2 <= SL_S <= 5 & 15 <= HCP <= 17 & Total_Points <= 18' where SL is suit length.

There are around 500K+ sequences. These 500K+ sequences are not yet curated. Curation will require the use of both expert human judgement and development of validation/coverages software.

## Related Projects
For a list of related projects see: https://github.com/BSalita/BridgeStats
