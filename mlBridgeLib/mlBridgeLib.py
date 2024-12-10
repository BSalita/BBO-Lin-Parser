# todo:
# use infer_types() or convert_dtypes() in _InsertScoringColumns instead of explicitly stating?
# make into class?
# create a class, move loose statements into __init__()
# implement assert statements that check function args for correctness, tuple count and len.
# remove unused functions.
# change functions that require dataframe to use list (series) instead.
# move dataframe functions to another source file?
# remove dependencies on: np, sklearn
# create validation functions for DDmakes, Hands, LoTT, HCP, dtypes, Vul, Dealer, Score, etc.


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # or DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def print_to_log_info(*args):
    print_to_log(logging.INFO, *args)
def print_to_log_debug(*args):
    print_to_log(logging.DEBUG, *args)
def print_to_log(level, *args):
    logging.log(level, ' '.join(str(arg) for arg in args))

import numpy as np
import pandas as pd
import polars as pl
import os
import pathlib
from collections import defaultdict
import re
from sklearn import preprocessing
import matplotlib.pyplot as plt
#from IPython.display import display  # needed for VSCode

# for double and single dummy calculations
from endplay.dealer import generate_deals
from endplay.types import Deal
from endplay.dds import par, calc_all_tables

# declare module read-only variables
CDHS = 'CDHS' # string ordered by suit rank - low to high
CDHSN = CDHS+'N' # string ordered by strain
NSHDC = 'NSHDC' # order by highest score value. useful for idxmax(). coincidentally reverse of CDHSN.
SHDC = 'SHDC' # Hands, PBN, board_record_string (brs) ordering
NSEW = 'NSEW' # double dummy solver ordering
NESW = 'NESW' # Hands and PBN order
NWES = 'NWES' # board_record_string (brs) ordering
SHDCN = 'SHDCN' # ordering used by dds
NextPosition = {'N':'E','E':'S','S':'W','W':'N'}
position_rotations_by_dealer = {'N':'NESW','E':'ESWN','S':'SWNE','W':'WNES'}
direction_order = NESW
NS_EW = ['NS','EW'] # list of partnership directions
major_suits = 'HS'
minor_suits = 'CD'
suit_order = CDHS
ranked_suit = 'AKQJT98765432' # card denominations - high to low
ranked_suit_rev = reversed(ranked_suit) # card denominations - low to high
ranked_suit_dict = {c:n for n,c in enumerate(ranked_suit)}
max_bidding_level = 7
tricks_in_a_book = 6
# todo: rename to VulToDDSVul
vul_d = {'None':0, 'Both':1, 'N_S':2, 'E_W':3} # dds vul encoding is weird
vul_syms = ['None','N_S','E_W','Both']
vul_directions = [[],[0,2],[1,3],[0,1,2,3]]
contract_types = ['Pass','Partial','Game','SSlam','GSlam']
dealer_d = {'N':0, 'E':1, 'S':2, 'W':3}
seats = ['N','E','S','W','NS','EW'] # Par score player's direction codes
PlayerDirectionToPairDirection = {'N':'NS','E':'EW','S':'NS','W':'EW'}
PairDirectionToOpponentPairDirection = {'NS':'EW','EW':'NS'}
allContracts = [(0,'Pass')]+[(l+1,s) for l in range(0,7) for s in CDHSN]
allHigherContracts_d = {c:allContracts[n+1:] for n,c in enumerate(allContracts)}
suit_names_d = {'S':'Spades','H':'Hearts','D':'Diamonds','C':'Clubs','N':'No-Trump'}
contract_classes = ['PASS'] + [level+suit+dbl+decl for level in '1234567' for suit in 'CDHSN' for dbl in ['','X','XX'] for decl in 'NESW']
contract_classes_dtype = pd.CategoricalDtype(contract_classes, ordered=False)
declarer_direction_to_pair_direction = {'N':'NS','S':'NS','E':'EW','W':'EW'}
# creates a dict all possible opening bids in auction order. key is npasses and values are opening bids.
auction_order = [level+suit for level in '1234567' for suit in 'CDHSN']+['x','xx','p'] # todo: put into mlBridgeLib

def pd_options_display():
    # display options overrides
    pd.options.display.max_columns = 0
    pd.options.display.max_colwidth = 100
    pd.options.display.min_rows = 500
    pd.options.display.max_rows = 10 # 0 is unlimited. 10 will use head(10/2) and tail(10/2).
    pd.options.display.precision = 2
    pd.options.display.float_format = '{:.2f}'.format

    # Don't wrap repr(DataFrame) across additional lines
    pd.options.display.expand_frame_repr = False
    #pd.set_option("display.expand_frame_repr", False)

#def beep(): # todo: jupyter notebooks and windows only. disable for others.
#    !powershell "[console]::beep(500,300)"

# todo: could save a couple seconds by creating dict of deals
def calc_double_dummy_deals(deals, batch_size=40):
    t_t = []
    tables = []
    for b in range(0,len(deals),batch_size):
        batch_tables = calc_all_tables(deals[b:min(b+batch_size,len(deals))])
        tables.extend(batch_tables)
        batch_t_t = (tt._data.resTable for tt in batch_tables)
        t_t.extend(batch_t_t)
    assert len(t_t) == len(tables)
    return deals, t_t, tables


def append_double_dummy_results(df):

    deals = [Deal(pbn) for pbn in df['PBN']]
    deals, t_t, tables = calc_double_dummy_deals(deals)

    # create df of pars
    # ParList attributes: score, default Contract[]
    # contract attributes: level, denom, declarer, penalty, result, from_auction(a,b), is_passout(), score(0)
    pars = [par(tt, b, 0) for tt,b in zip(tables,df['Board'])] # middle arg is board (if int) otherwise enum vul.
    par_scores = [parlist.score for parlist in pars]
    par_df = pd.DataFrame(par_scores,columns=['ParScore_EndPlay_NS'])
    par_df['ParScore_EndPlay_EW'] = -par_df['ParScore_EndPlay_NS']
    par_contracts = [[str(contract.level)+SHDCN[int(contract.denom)]+contract.declarer.abbr+contract.penalty.abbr+' '+str(contract.result) for contract in parlist] for parlist in pars]
    par_df['ParContracts_EndPlay'] = par_contracts

    rows = []
    direction_order = [0,2,1,3] # NSEW order
    suit_order = [3,2,1,0,4] # SHDCN order?
    for ii,(dd,sd,tt,ps,pc) in enumerate(zip(deals,t_t,tables,par_scores,par_contracts)):
        # dd.pprint() # display deal
        # print()
        # tt.pprint() # display double dummy table
        # print_to_log()
        # print_to_log(ps)
        # print_to_log()
        # print_to_log(pc)
        nsew_flat_l = [sd[suit][direction] for direction in direction_order for suit in suit_order]
        rows.append(nsew_flat_l)
    assert len(rows) == len(df)
    dd_df = pd.DataFrame(rows,columns=['_'.join(['DD',d,s]) for d in NSEW for s in CDHSN])
    df = pd.concat([df,dd_df,par_df],axis='columns') # be sure df's index is reset before concat.
    return df


def constraints(deal):
    return True


def generate_single_dummy_deals(predeal_string, produce, env=dict(), max_attempts=1000000, seed=None, show_progress=True, strict=True, swapping=0):
    
    predeal = Deal(predeal_string)

    deals_t = generate_deals(
        constraints,
        predeal=predeal,
        swapping=swapping,
        show_progress=show_progress,
        produce=produce,
        seed=seed,
        max_attempts=max_attempts,
        env=env,
        strict=strict
        )

    deals = tuple(deals_t) # create a tuple before interop memory goes wonky
    
    return calc_double_dummy_deals(deals)


def calculate_single_dummy_probabilities(deal, produce=100):

    ns_ew_rows = {}
    for ns_ew in ['NS','EW']:
        s = deal[2:].split()
        if ns_ew == 'NS':
            s[1] = '...'
            s[3] = '...'
        else:
            s[0] = '...'
            s[2] = '...'
        predeal_string = 'N:'+' '.join(s)
        #print_to_log(f"predeal:{predeal_string}")

        d_t, t_t, tables = generate_single_dummy_deals(predeal_string, produce, show_progress=False)

        rows = []
        max_display = 4 # pprint only the first n generated deals
        direction_order = [0,2,1,3] # NSEW order
        suit_order = [3,2,1,0,4] # SHDCN order?
        for ii,(dd,sd,tt) in enumerate(zip(d_t,t_t,tables)):
            # if ii < max_display:
                # print_to_log(f"Deal:{ii+1} Fixed:{ns_ew} Generated:{ii+1}/{produce}")
                # dd.pprint()
                # print_to_log()
                # tt.pprint()
                # print_to_log()
            nswe_flat_l = [sd[suit][direction] for direction in direction_order for suit in suit_order]
            rows.append([dd.to_pbn()]+nswe_flat_l)

        dd_df = pd.DataFrame(rows,columns=['Deal']+[d+s for d in NSEW for s in CDHSN])
        for d in NSEW:
            for s in SHDCN:
                ns_ew_rows[(ns_ew,d,s)] = dd_df[d+s].value_counts(normalize=True).reindex(range(14), fill_value=0).tolist() # ['Fixed_Direction','Direction_Declarer','Suit']+['SD_Prob_Take_'+str(n) for n in range(14)]
    
    return ns_ew_rows


def append_single_dummy_results(pbns,sd_cache_d,produce=100):

    for pbn in pbns:
        if pbn not in sd_cache_d:
            sd_cache_d[pbn] = calculate_single_dummy_probabilities(pbn, produce) # all combinations of declarer pair direction, declarer direciton, suit, tricks taken
    return sd_cache_d


def sort_suit(s):
    return '' if s == '' else ''.join(sorted(s,key=lambda c: ranked_suit_dict[c]))


def sort_hand(h):
    return [sort_suit(s) for s in h]


# Create a board_record_string from hand.
#def HandToBoardRecordString(hand):
#    return ''.join([s+c for h in [hand[0],hand[3],hand[1],hand[2]] for s,c in zip('SHDC',h)])


# Create a tuple of suit lengths per direction (NESW).
# todo: assert that every suit has 13 cards

def HandsToSuitLengths(hands):
    t = tuple(HandToSuitLengths(hand) for hand in hands)
    assert sum(h[0] for h in t) == 52
    return sum(h[0] for h in t),t
def HandToSuitLengths(hand):
    t = tuple(SuitToSuitLengths(suit) for suit in hand)
    return sum(t),t
def SuitToSuitLengths(suit):
    return len(suit)


# Calculate distribution points using 3-2-1 system.
dist_points = [3,2,1]+[0]*11 # using traditional distribution points metric
assert len(dist_points) == 14 # 14 possible suit lengths
def HandsToDistributionPoints(hands):
    t = tuple(HandToDistributionPoints(hand) for hand in hands)
    return sum(h[0] for h in t),t
def HandToDistributionPoints(hand):
    t = tuple(SuitToDistributionPoints(suit) for suit in hand)
    return sum(t),t
def SuitToDistributionPoints(suit):
    return dist_points[len(suit)]


# Create a tuple of suit lengths per direction (NESW).
# todo: assert that every suit has 13 cards
# def CombinedSuitLengths(h):
#     t = SuitLengths(h)
#     return tuple(tuple(sp1+sp2 for sp1,sp2 in zip(h1,h2)) for h1,h2 in [[t[0],t[2]],[t[1],t[3]]])


# Create a tuple of combined suit lengths (NS, EW) sorted by largest to smallest
#def SortedSuitLengthTuples(h):
#    t = CombinedSuitLengths(h)
#    return tuple((t[i],i,'SHDC'[i]) for (v, i) in sorted([(v, i) for (i, v) in enumerate(t)],reverse=True))


def hrs_to_brss(hrs,void='',ten='10'):
    cols = [d+'_'+s for d in ['north','west','east','south'] for s in ['spades','hearts','diamonds','clubs']] # remake of hands below, comments says the order needs to be NWES?????
    return hrs[cols].apply(lambda r: ''.join(['SHDC'[i%4]+c for i,c in enumerate(r.values)]).replace(' ','').replace('-',void).replace('10',ten), axis='columns')


# board_record_string (brs) is NWES, SHDC order
# pbn is NESW, SHDC order
# hands is NESW, SHDC order

def pbn_to_brs(pbn,void='',ten='10'):
    r = [r'(.*)\.(.*)\.(.*)\.(.*)']
    rs = r'^N\:'+' '.join(r*4)+r'$'
    nesw = [shdc+(hand if len(hand) else void) for shdc,hand in zip(SHDC*4,re.match(rs,pbn).groups())] # both use SHDC order
    return ''.join([''.join(nesw[i*4:i*4+4]) for i in [0,3,1,2]]).replace('T',ten) # pbn uses NESW order but we want NWES


def pbn_to_hands(pbn):
    r = [r'(.*)\.(.*)\.(.*)\.(.*)']
    rs = r'^N\:'+' '.join(r*4)+r'$'
    return tuple([tuple([hand for hand in re.match(rs,pbn).groups()[i*4:i*4+4]]) for i in range(4)]) # both use NESW order


def validate_brs(brs):
    assert '-' not in brs and 'T' not in brs, brs # must not have a '-' or 'T'
    sorted_brs = '22223333444455556666777788889999AAAACCCCDDDDHHHHJJJJKKKKQQQQSSSSTTTT' # sorted brs must match this string
    s = brs.replace('10','T')
    if ''.join(sorted(s)) != sorted_brs:
        print_to_log_info('validate_brs: Invalid brs:', brs, s)
        return False
    for i in range(0,len(sorted_brs),len(sorted_brs)*4):
        split_shdc = re.split(r'[SHDC]',s[i:i+13+4])
        if len(split_shdc) != 4+1 or sum(map(len,split_shdc)) != 13: # not validating sort order. call it correct-ish.
            print_to_log_info('validate_brs: Invalid len:', i, brs, s[i:i+13+4], split_shdc)
            return False
    return True


def LinToPBN(df):
    return ['N:'+' '.join(['.'.join(list(map(lambda x: x[::-1], re.split('S|H|D|C', hh)))[1:]) for hh in r]) for r in df.select(pl.col(r'^Hand_[NESW]$')).rows()]
    

def brs_to_pbn(brs,void='',ten='T'):
    r = r'S(.*)H(.*)D(.*)C(.*)'
    rs = r*4
    suits = [suit for suit in re.match(rs,brs).groups()]
    return 'N:'+' '.join(['.'.join(suits[i*4:i*4+4]) for i in [0,2,3,1]]).replace('10',ten).replace('-',void) # brs uses NWES order but we want NESW. void may or not contain '-'


def brs_to_hands(brs,void='',ten='T'):
    no_10s = brs.replace('10',ten).replace('-',void) # replace 10 with T and remove unnecessary '-' which signifies a void suit.
    assert len(no_10s) == (13+len('SHDC'))*4 # (13 cards per suit + 4 suit symbols) * 4
    nesw = tuple([no_10s[i:i+17] for i in range(0,17*4,17)])
    assert len(nesw) == 4 and all(len(s) == 17 for s in nesw), [nesw, brs]
    return tuple([brs_to_hand(nesw[i]) for i in [0,2,3,1]]) # brs uses NWES order but we want NESW


def brs_to_hand(brs):
    assert isinstance(brs,str) and len(brs) == 17, brs
    split_shdc = re.split(r'[SHDC]',brs) # returns a tuple of 5
    assert len(split_shdc) == 4+1 and sum(map(len,split_shdc)) == 13, brs
    return tuple(sort_hand(split_shdc[1:]))


def hands_to_brs(hands,void='',ten='10'):
    brs = ''.join([c+(suit if len(suit) else void) for i in [0,3,1,2] for c,suit in zip(SHDC,hands[i])]).replace('T',ten) # hands uses NESW order but we want NWSE.
    return brs


# convert hand tuple into PBN
def HandToPBN(hand):
    assert len(hand) == 4 and all([sum(map(len,hand)) == 16 for h in hand]), hand # 4 hands of 4 suits. 13 cards per hand + 3 dots == 16
    return 'N:'+' '.join('.'.join([suit for suit in hand[i]]) for i in range(4)) # both use NESW order.


# create list of PBNs from Hands (list of tuple of tuples)
def HandsToPBN(hands):
    pbns = []
    for hand in hands:
        pbns.append(HandToPBN(hand)) # both use NESW order.
    return pbns


# Create tuple of suit lengths per partnership (NS, EW)
#def CombinedSuitLengthTuples(t):
#    return tuple([tuple([sn+ss for sn,ss in zip(t[0],t[2])]),tuple([se+sw for se,sw in zip(t[1],t[3])])])


hcpd = {c:w for c,w in zip(ranked_suit,[4,3,2,1]+[0]*9)}
def HandsToHCP(hands):
    t = tuple(HandToHCP(hand) for hand in hands)
    assert sum(h[0] for h in t) == 40
    return sum(h[0] for h in t),t
def HandToHCP(hand):
    t = tuple(SuitToHCP(suit) for suit in hand)
    return sum(t),t
def SuitToHCP(suit):
    return sum(hcpd[c] for c in suit)


# Convert list of tupled hands to binary string.
def HandsLToBin(handsl):
    return [tuple(hand[0] for hand in HandsToBin(hands)) for hands in handsl]


# Convert list of hands, in binary string format, to One Hot Encoded list
def BinLToOHE(binl):
    return [tuple((int(i) for hand in hands for i in f'{hand[2:].zfill(52)}')) for hands in binl]


# Convert One Hot Encoded hands to tupled hands.
def OHEToHandsL(ohel):
    return [tuple(tuple(([''.join([ranked_suit[denom] for denom in range(13) if hands[hand+suit*13+denom]]) for suit in range(4)])) for hand in range(0,52*4,52)) for hands in ohel]


# Convert One Hot Encoded hands to cards.
def OHEToCards(df, ohel):
    return pd.DataFrame(ohel,index=df.index,columns=['C_'+nesw+suit+denom for nesw in NESW for suit in SHDC for denom in ranked_suit],dtype='int8')


# Create column of binary encoded hands
wd = {c:1<<n for n,c in enumerate(ranked_suit_rev)}
def HandsToBin(hands):
    t = tuple(HandToBin(hand) for hand in hands)
    assert sum(tt[0] for tt in t) == (1<<(13*4))-1
    return tuple(tuple([bin(h[0])[2:].zfill(52),tuple(bin(s)[2:].zfill(13) for s in h[1])]) for h in t)
def HandToBin(hand):
    t = tuple(SuitToBin(suit) for suit in hand)
    tsum = sum(h<<(n*13) for n,h in enumerate(reversed(t))) # order spades to clubs
    return tsum,t
def SuitToBin(suit):
    return sum([wd[c] for c in suit])


# Create column of hex encoded hands
def HandsToHex(hands):
    t = tuple(HandToHex(hand) for hand in hands)
    assert sum(tt[0] for tt in t) == (1<<(13*4))-1
    return tuple(tuple([hex(h[0]),tuple(hex(s) for s in h[1])]) for h in t)
def HandToHex(hand):
    t = tuple(SuitToHex(suit) for suit in hand)
    tsum = sum(h<<(n*13) for n,h in enumerate(reversed(t))) # order spades to clubs
    return tsum,t
def SuitToHex(suit):
    return sum([wd[c] for c in suit])


# Create column of Quick Trick values. Might be easier to do using binary encoded hands.
qtl = [(2,'AK'),(1.5,'AQ'),(1,'A'),(1,'KQ'),(0.5,'K')] # list of (quick tricks card combos, combo value)
qtls = sorted(qtl,reverse=True) # sort by quick trick value (most to least) to avoid ambiguity
def HandsToQT(hands):
    t = tuple(HandToQT(hand) for hand in hands)
    return sum(h[0] for h in t),t
def HandToQT(hand):
    t = tuple(SuitToQT(suit) for suit in hand)
    return sum(t),t
def SuitToQT(suit):
    # assumes suits are sorted by HCP value (most to least) (AKQJT...)
    for qt in qtls:
        if suit.startswith(qt[1]):
            return qt[0]
    return 0


def BoardNumberToDealer(bn):
    return NESW[(bn-1) & 3]


def BoardNumberToVul(bn):
    bn -= 1
    return range(bn//4, bn//4+4)[bn & 3] & 3


#def HandsToDDMakes(hands):
#    return tuple([tuple([[df['_'.join(['DD',d,s])] for s in 'SHDC']]) for d in NESW])


# create column of LoTT.
# todo: verify algorithm against actual LoTT.
# I'm confused about the order of the suits and lengths. What about a tie between max suits. It should use highest ranking suit. 
# Are all callers really passing dd ordered CDHSN/SHDC, and SL ordered CDHS?
# Renamed to LoTT_SHDC until verified.
# Callers should use a dict so LoTT isn't recomputed for every board result. Only cache by board, not by board result.
def LoTT_SHDC(ddmakes,lengths):
    maxnsl = []
    maxewl = []
    for nsidx,(nmakes,smakes,nlength,slength) in enumerate(zip(ddmakes[0][:4][::-1],ddmakes[2][:4][::-1],lengths[1][0][1],lengths[1][2][1])): # [::-1] to reverse ddmakes
        nsmax = max(nmakes,smakes)
        maxnsl.append((nlength+slength,nsmax,nsidx))
    for ewidx,(emakes,wmakes,elength,wlength) in enumerate(zip(ddmakes[1][:4][::-1],ddmakes[3][:4][::-1],lengths[1][1][1],lengths[1][3][1])): # [::-1] to reverse ddmakes
        ewmax = max(emakes,wmakes)
        maxewl.append((elength+wlength,ewmax,ewidx))
    sorted_maxnsl = sorted(maxnsl,reverse=True)
    sorted_maxewl = sorted(maxewl,reverse=True)
    maxlen = sorted_maxnsl[0][0]+sorted_maxewl[0][0]
    maxmake = sorted_maxnsl[0][1]+sorted_maxewl[0][1]
    return (maxmake,maxlen,maxmake-maxlen)


def ContractType(tricks,suit):
    if tricks is None or tricks < 7:
        ct = 'Pass'
    elif tricks == 12:
        ct = 'SSlam'
    elif tricks == 13:
        ct = 'GSlam'
    elif suit in 'CD' and tricks in range(11,12):
        ct = 'Game'
    elif suit in 'HS' and tricks in range(10,12):
        ct = 'Game'
    elif suit in 'N' and tricks in range(9,12):
        ct = 'Game'
    else:
        ct = 'Partial'
    return ct


def ContractTypeFromContract(contract):
    # contact is 'PASS'|[1-7][CDHSN]X*[NESW]
    if contract[0] == 'P':
        tricks = 0
        suit = None
    else:
        tricks = int(contract[0])+6
        suit = contract[1]
    return ContractType(tricks,suit)


def CategorifyContractTypeBySuit(ddmakes):
    contract_types_d = defaultdict(list)
    for dd in ddmakes:
        for direction,nesw in zip(NS_EW,dd): # todo: using NS_EW instead of NESW for now. switch to NESW?
            for suit,tricks in zip(CDHSN,nesw):
                assert tricks is not None
                ct = ContractType(tricks,suit)
                contract_types_d['_'.join(['CT',direction,suit])].append(ct) # estimators don't like categorical dtype
    return contract_types_d


# Create columns of contract type booleans by partnership by suit by contract. e.g. CT_NS_C_Game
def CategorifyContractTypeByDirection(df):
    contract_types_d = {}
    cols = df.select(pl.col(r'^CT_(NS|EW)_[CDHSN]$')).columns
    for c in cols:
        for t in contract_types:
            #print_to_log_debug('CT:',c,t)
            new_c = c+'_'+t
            contract_types_d[new_c] = (t == df[c])
    return contract_types_d


# convert vul to boolean based on direction
# todo: rename to DirectionVulToBool
def DirectionToVul(vul, nesw):
    return nesw in vul_directions[vul_syms.index(vul)]


# convert vul number (0,3) to boolean based on direction
def DirectionSymToVulBool(vul,nsew):
    return NESW.index(nsew) in vul_directions[vul]


# convert board to vul number (0,3] to boolean based on direction
def BoardNumberDirectionSymToVulBool(board,nsew):
    return DirectionSymToVulBool(BoardNumberToVul(board),nsew)


def DirectionSymToDealer(direction_symbol):
    return list(NESW).index(direction_symbol) # using NSEW index because of score()


def StrainSymToValue(strain_symbol):
    return list(CDHSN).index(strain_symbol) # using CDHSN index because of score()


# Create list of tuples of (score, (level, strain), direction, result). Useful for calculating Pars.
# todo: rewrite into two defs; looping, core logic
def DDmakesToScores(ddmakes,vuls):
    scoresl = []
    for ddmakes,vul in zip(ddmakes,vuls):
        directionsl = []
        for direction in range(len(NESW)):
            # todo: add to mlBridgeLib
            v =  DirectionToVul(vul,direction)
            strainl = []
            for strain, tricks in enumerate(ddmakes[direction]): # cycle through all strains
                highest_make_level = tricks-1-tricks_in_a_book
                for level in range(max(highest_make_level,0), max_bidding_level):
                    result = highest_make_level-level
                    s = score(level, strain, result < 0, 0, v, result) # double all sets
                    strainl.append((s,(level,strain),direction,result))
            # stable sort by contract then score
            sorted_direction = sorted(sorted(strainl,key=lambda k:k[1]),reverse=True,key=lambda k:k[0])
            directionsl.append(sorted_direction)
        scoresl.append(directionsl)
    return scoresl

def ContractToScores(df,direction='Declarer_Direction',cache={}):
    assert 'NSEW' not in df and direction in df
    scores_l = []
    for bidlvl,bidsuit,direction,ivul,dbl in df[['BidLvl','BidSuit',direction,'iVul','Dbl']].to_numpy(): # rows: # convert df to list of tuples
        if bidlvl is pd.NA or bidlvl == 0: # Contract of 'PASS' in which case BidSuit and Dbl are nulls.
            scores = [[0]*14]
        elif (bidlvl,bidsuit,direction,ivul,dbl) in cache:
            scores = cache[(bidlvl,bidsuit,direction,ivul,dbl)]
        else:
            scores = scoresd[bidlvl-1,StrainSymToValue(bidsuit),DirectionSymToDealer(direction) in vul_directions[ivul],len(dbl),'NSEW'.index(direction)]
            cache[(bidlvl,bidsuit,direction,ivul,dbl)] = scores
        scores_l.append(scores)
    # what to do about adjusted scores? assert df['Score_NS'].isin(scores_l).all(), df[df.apply(lambda r: r['Score_NS'] not in r['scores_l'],axis='columns')]
    return scores_l

# Convert score tuples into Par.
# todo: rewrite into two defs; looping, core logic
# todo: Is this still working? Compare against dds generated pars.
def ScoresToPar(scoresl):
    par_scoresll = []
    for directionsl in scoresl: # [scoresl[0]]:
        par_scoresl = [(0,(0,0),0,0)]
        direction = 0
        while(True):
            d_ew = direction & 1
            #print_to_log(directionsl[direction])
            for par_score in directionsl[direction]: # for each possible remaining bid
                if par_scoresl[0][1] < par_score[1]: # bid is sufficient
                    #print_to_log("suff:",par_scoresl[0],par_score)
                    psl_ew = par_scoresl[0][2] & 1
                    #print_to_log(direction,d_ew,ps_ew,((direction ^ ps_ew) & 1),((d_ew ^ ps_ew) & 1))
                    assert ((direction ^ psl_ew) & 1) == ((d_ew ^ psl_ew) & 1)
                    opponents = d_ew != psl_ew
                    assert (d_ew != psl_ew) == opponents
                    if opponents:
                        #print_to_log("oppo:",-par_scoresl[0][0],par_score[0],d_ew,ps_ew)
                        if -par_scoresl[0][0] <= par_score[0]: # bidder was opponent, improved score is a sacrifice
                            par_scoresl.insert(0,par_score)
                            #error
                            #break
                        else:
                            break
                    else:
                        if par_scoresl[0][0] <= par_score[0]: # bidder was partnership, take improved score
                            #print_to_log("same:",par_scoresl[0][0],par_score[0],len(par_scoresl))
                            par_scoresl.insert(0,par_score)
                            #break
                        else:
                            break
            direction = (direction+1) % len(NESW)
            #print_to_log(direction,par_scoresl[0][2])
            if direction == par_scoresl[0][2]: # bidding is over when new direction is last bidder
                break
        parl = []
        score = par_scoresl[0][0]
        psl_ew = par_scoresl[0][2] & 1
        par_scores_formatted = (-score if psl_ew else score,parl)
        for par_score in par_scoresl:
            ps_ew = par_score[2] & 1
            if len(parl) > 0 and (score != par_score[0] or psl_ew != ps_ew): # only use final score in same direction
                break
            result = par_score[3]
            par = tuple((par_score[1][0]+1,CDHSN[par_score[1][1]],'*' if result < 0 else '',['NS','EW'][ps_ew],result))
            parl.insert(0,par)
        #display(par_scores_formatted)
        par_scoresll.append(par_scores_formatted)
    return par_scoresll


# todo: don't pass row. pass only necessary values
#def LoTT(r):
#    t = []
#    for d in range(0,2):
#        max_suit_length_tuple = r['Suit_Lengths_Sorted'][d][0]
#        suit_length, suit_idx, suit_char = max_suit_length_tuple
#        dd_makes_suit_idx = 3-suit_idx
#        dd_makes = r['DDmakes'][d][dd_makes_suit_idx],r['DDmakes'][d+2][dd_makes_suit_idx]
#        variance = suit_length-dd_makes[0],suit_length-dd_makes[1]
#        t.append(tuple([suit_char,suit_length,dd_makes,variance]))
#    return tuple(t)


def FilterBoards(df, cn=None, vul=None, direction=None, suit=None, contractType=None, doubles=None):
    # optionally filter dataframe's rows
    if not cn is None:
        # only allow this club number e.g. not subclubs 108571/267096
        df = df[df['Key'].str.startswith(cn)]
    if not vul is None:
        # one of the following: 'None','NS','EW','Both'
        if vul == 'None':
            df = df[~(df['Vul_NS'] | df['Vul_EW'])]  # neither NS, EW
        elif vul == 'NS':
            df = df[df['Vul_NS']]  # only NS
        elif vul == 'EW':
            df = df[df['Vul_EW']]  # only EW
        elif vul == 'Both':
            df = df[df['Vul_NS'] & df['Vul_NS']]  # only Both
        else:
            print_to_log_info(f'FilterBoards: Error: Invalid vul:{vul}')
    if not direction is None:
        # either 'NS','EW' # Single direction is problematic so using NS, EW
        df = df[df['Par_Dir'] == direction]
    if not suit is None:
        df = df[df['Par_Suit'].isin(suit)]  # ['CDHSN']
    if not contractType is None:
        # ['Pass',Partial','Game','SSlam','GSlam']
        df = df[df['Par_Type'].isin(contractType)]
    if not doubles is None:
        # ['','*','**'] # Par scores only are down if they're sacrifices
        df = df[df['Par_Double'].isin(doubles)]
    df.reset_index(drop=True, inplace=True)
    return df


# adapted (MIT license) from https://github.com/jfklorenz/Bridge-Scoring/blob/master/features/score.js
# ================================================================
# Scoring
def score(level, suit, double, declarer, vulnerability, result, declarer_score=False):
    assert level in range(0, 7), f'ValueError: level {level} is invalid'
    assert suit in range(0, 5), f'ValueError: suit {suit} is invalid' # CDHSN
    assert double in range(0, 3), f'ValueError: double {double} is invalid' # ['','X','XX']
    assert declarer in range(
        0, 4), f'ValueError: declarer {declarer} is invalid' # NSEW
    assert vulnerability in range(
        0, 2), f'ValueError: vulnerability {vulnerability} is invalid'
    assert result in range(-13, 7), f'ValueError: result {result} is invalid'

    # Contract Points
    points_contract = [
        [[20, 40, 80], [20, 40, 80]],
        [[20, 40, 80], [20, 40, 80]],
        [[30, 60, 120], [30, 60, 120]],
        [[30, 60, 120], [30, 60, 120]],
        [[40, 80, 160], [30, 60, 120]]
    ]

    # Overtrick Points
    overtrick = [
        [[20, 100, 200], [20, 200, 400]],
        [[20, 100, 200], [20, 200, 400]],
        [[30, 100, 200], [30, 200, 400]],
        [[30, 100, 200], [30, 200, 400]],
        [[30, 100, 200], [30, 200, 400]]
    ]

    # Undertrick Points
    undertricks = [
        [[50, 50, 50, 50], [100, 200, 200, 300], [200, 400, 400, 600]],
        [[100, 100, 100, 100], [200, 300, 300, 300], [400, 600, 600, 600]]
    ]

    # Bonus Points
    bonus_game = [[50, 50], [300, 500]]
    bonus_slam = [[500, 750], [1000, 1500]]
    bonus_double = [0, 50, 100]

    if result >= 0:
        points = points_contract[suit][0][double] + \
            level * points_contract[suit][1][double]

        points += bonus_game[points >= 100][vulnerability]

        if level >= 5:
            points += bonus_slam[level - 5][vulnerability]

        points += bonus_double[double] + result * \
            overtrick[suit][vulnerability][double]

    else:
        points = -sum([undertricks[vulnerability][double][min(i, 3)]
                       for i in range(0, -result)])

    return points if declarer_score or declarer < 2 else -points  # negate points if EW

# ================================================================


# create some helpful scoring dicts
# de-cumsum set scores
def ScoreUnderTricks(level, suit, double, declarer, vulnerability):
    l = [score(level, suit, double, declarer, vulnerability, result)
         for result in list(range(-7-level, 0))]
    return [s-ss for s, ss in zip(l, l[1:]+[0])]+[0]*(7-level)


# de-cumsum make scores
def ScoreOverTricks(level, suit, double, declarer, vulnerability):
    l = [score(level, suit, double, declarer, vulnerability, result)
         for result in list(range(0, 7-level))]
    return [0]*(7+level)+[s-ss for s, ss in zip(l, [0]+l[:-1])]


def ScoreDicts():
    # Returns 3 dicts useful for scoring. Each dict expects a tuple: (level, suit, vulnerability, double, declarer)
    # Examples:
    #   scoresd[(0,0,0,0,0)] return [-350, -300, -250, -200, -150, -100, -50, 70, 90, 110, 130, 150, 170, 190]
    #   makeScoresd[(0,0,0,0,0)] return [0, 0, 0, 0, 0, 0, 0, 70, 20, 20, 20, 20, 20, 20]
    #   setScoresd[(0,0,0,0,0)] return [-50, -50, -50, -50, -50, -50, -50, 0, 0, 0, 0, 0, 0, 0]    
    scoresd = {(level, suit, vulnerability, double, declarer): [score(level, suit, double, declarer, vulnerability, result) for result in list(range(-7-level, 0))+list(
        range(0, 7-level))] for declarer in range(0, 4) for level in range(0, 7) for suit in range(0, 5) for double in range(0, 3) for vulnerability in range(0, 2)}
    assert sum([len(v) != 14 for v in scoresd.values()]) == 0
    # display(scoresd)
    setScoresd = {(level, suit, vulnerability, double, declarer): ScoreUnderTricks(level, suit, double, declarer, vulnerability)
                  for declarer in range(0, 4) for level in range(0, 7) for suit in range(0, 5) for double in range(0, 3) for vulnerability in range(0, 2)}
    # display(setScoresd)
    assert sum([len(v) != 14 for v in setScoresd.values()]) == 0
    makeScoresd = {(level, suit, vulnerability, double, declarer): ScoreOverTricks(level, suit, double, declarer, vulnerability)
                   for declarer in range(0, 4) for level in range(0, 7) for suit in range(0, 5) for double in range(0, 3) for vulnerability in range(0, 2)}
    # display(makeScoresd)
    assert sum([len(v) != 14 for v in makeScoresd.values()]) == 0
    return scoresd, setScoresd, makeScoresd


# returns scoring array(14) where all sets are doubled. For situations where perfect defense is assumed (Par, match point predictions)
def ScoreDoubledSets(level, suit, vul, double, declarer): # vul = is declarer vul? declarer = NSEW.index(declarer). if declarer is EW (declarer >= 2), returns negative.
    return scoresd[(level, suit, vul, 1, declarer)][:level+7]+scoresd[(level, suit, vul, double, declarer)][7+level:]


# insert Actual and Predicted as leftmost columns for easier viewing.
# default column creation is rightmost
def MakeColName(prefix, name, suit, direction):
    if name != '':
        name = '_'+name
    if suit != '':
        suit = '_'+suit
    if direction != '':
        direction = '_'+direction
    return prefix+name+suit+direction


def MakeSuitCols(prefix, suit, direction):
    return ['_'.join([prefix, suit, direction])+str(n).zfill(2) for n in range(0, 14)]


def AssignToColumn(df, name, values, dtype=None):
    df[name] = values
    if dtype is not None:
        df[name] = df[name].astype(dtype)  # float doesn't support pd.NA
    return df[name]


def AssignToColumnLoc(df, bexpr, name, values, dtype=None):
    df.loc[bexpr, name] = values
    if dtype is not None:
        df[name] = df[name].astype(dtype)  # float doesn't support pd.NA
    return df.loc[bexpr, name]


def InsertTcgColumns(df, dep_vars, prefix, tcgd, tcg299d):
    dep_var, new_dep_var, suit, direction, double = dep_vars
    # colnum = len(df.columns)-1 # todo: subtract one until colnum+1 is adjusted in inserts
    # create TCG_Key for indexing into tcgd to obtain common game data. TCG_Key has no direction.
    bnotna = df[MakeColName(prefix, 'Score', suit, direction)].notna()
    # todo: Is this the best place for replace('E2A','A')?
    AssignToColumn(df, MakeColName(prefix, 'TCG_Key', suit, direction), df.loc[bnotna, 'EventBoard'].str.replace(
        'E2A', 'A').str.cat(df.loc[bnotna, MakeColName(prefix, 'Score', suit, direction)].astype(int).map(str), sep='_'), 'string')
    # TCG stuff returns NS MP, never EW.
    tcgns = GetTcgMPs(tcgd, df[MakeColName(
        prefix, 'TCG_Key', suit, direction)])
    tcg299ns = GetTcgMPs(
        tcg299d, df[MakeColName(prefix, 'TCG_Key', suit, direction)])
    if direction == '' or direction == 'NS':
        AssignToColumn(df, MakeColName(
            prefix, 'TCG_MP', suit, 'NS'), tcgns, 'float')
        AssignToColumn(df, MakeColName(
            prefix, 'TCG299_MP', suit, 'NS'), tcg299ns, 'float')
    if direction == '' or direction == 'EW':
        tcgew = [1-mp for mp in tcgns]
        tcg299ew = [1-mp for mp in tcg299ns]
        AssignToColumn(df, MakeColName(
            prefix, 'TCG_MP', suit, 'EW'), tcgew, 'float')
        AssignToColumn(df, MakeColName(
            prefix, 'TCG299_MP', suit, 'EW'), tcg299ew, 'float')
    return


def FormatBid(df, prefix, dep_vars):
    dep_var, new_dep_var, suit, direction, double = dep_vars
    # alternative: if MakeColName(prefix, 'Dir', suit, direction) in df.columns
    if direction == '':
        nsew = df[MakeColName(prefix, 'Dir', suit, direction)]
    else:
        nsew = direction
    return df[MakeColName(prefix, 'Level', suit, direction)].map(
        str)+df[MakeColName(prefix, 'Suit', suit, direction)]+df[MakeColName(prefix, 'Double', suit, direction)]+' '+nsew+' '+(df[MakeColName(prefix, 'Result', suit, direction)].map(str))


def InsertScoringColumnsPar(df, dep_vars, prefix):
    # df has 'Level' but not 'Tricks'
    dep_var, new_dep_var, suit, direction, double = dep_vars
    values = df[MakeColName(prefix, 'Level', suit, direction)]
    assert all(values <= 7), [v for v in values if v > 7]
    AssignToColumnLoc(df, values > 0, MakeColName(
        prefix, 'Tricks', suit, direction), values+6, 'Int8')
    return InsertScoringColumns(df, dep_vars, prefix)


def InsertScoringColumnsTricks(df, dep_vars, prefix):
    # df has 'Tricks' but not 'Level'
    dep_var, new_dep_var, suit, direction, double = dep_vars
    values = df[MakeColName(prefix, 'Tricks', suit, direction)]
    assert all(values >= 0) and all(values <= 13), [v for v in values if v < 0 or v > 13]
    AssignToColumnLoc(df, values > 6, MakeColName(
        prefix, 'Level', suit, direction), values-6, 'Int8')
    return InsertScoringColumnsSDR(df, dep_vars, prefix)


def InsertScoringColumnsSDR(df, dep_vars, prefix):
    dep_var, new_dep_var, suit, direction, double = dep_vars
    AssignToColumn(df, MakeColName(
        prefix, 'Suit', suit, direction), suit, 'string')
    AssignToColumn(df, MakeColName(prefix, 'Double',
                                   suit, direction), double, 'string')
    AssignToColumn(df, MakeColName(prefix, 'Result', suit, direction),
                   df[dep_var]-df[MakeColName(prefix, 'Tricks', suit, direction)], 'Int8')
    return InsertScoringColumns(df, dep_vars, prefix)


def InsertScoringColumns(df, dep_vars, prefix):
    dep_var, new_dep_var, suit, direction, double = dep_vars
    AssignToColumnLoc(df, df[MakeColName(prefix, 'Tricks', suit, direction)] >= 7, MakeColName(
        prefix, 'Bid', suit, direction), FormatBid(df, prefix, dep_vars), 'string')
    AssignToColumnLoc(df, df[MakeColName(prefix, 'Tricks', suit, direction)] < 7, MakeColName(
        prefix, 'Bid', suit, direction), 'Pass', 'string')  # not worth a bid

    # Calculate contract type
    AssignToColumn(df, MakeColName(
        prefix, 'Type', suit, direction), 'Pass', 'string')
    AssignToColumnLoc(df, df[MakeColName(prefix, 'Level', suit, direction)] > 0, MakeColName(
        prefix, 'Type', suit, direction), 'Partial', 'string')
    AssignToColumnLoc(df, (df[MakeColName(prefix, 'Suit', suit, direction)].isin(['C', 'D'])) & (
        df[MakeColName(prefix, 'Level', suit, direction)] >= 5), MakeColName(prefix, 'Type', suit, direction), 'Game', 'string')
    AssignToColumnLoc(df, (df[MakeColName(prefix, 'Suit', suit, direction)].isin(['H', 'S'])) & (
        df[MakeColName(prefix, 'Level', suit, direction)] >= 4), MakeColName(prefix, 'Type', suit, direction), 'Game', 'string')
    AssignToColumnLoc(df, (df[MakeColName(prefix, 'Suit', suit, direction)] == 'N') & (df[MakeColName(
        prefix, 'Level', suit, direction)] >= 3), MakeColName(prefix, 'Type', suit, direction), 'Game', 'string')
    AssignToColumnLoc(df, df[MakeColName(prefix, 'Level', suit, direction)] >= 6, MakeColName(
        prefix, 'Type', suit, direction), 'SSlam', 'string')
    AssignToColumnLoc(df, df[MakeColName(prefix, 'Level', suit, direction)] == 7, MakeColName(
        prefix, 'Type', suit, direction), 'GSlam', 'string')

    # Calculate Score
    scoresl = []
    for i, r in df.iterrows():
        s = r[MakeColName(prefix, 'Level', suit, direction)]
        if s is not pd.NA:  # Level is a dtype that supports pd.NA (int8)
            if s > 0:
                # Calculate vulnerability
                if direction == '':
                    idirection = NSEW.index(
                        r[MakeColName(prefix, 'Dir', suit, direction)][0])
                else:
                    idirection = NSEW.index(direction[0])
                vul = [r['Vul_NS'], r['Vul_NS'],
                       r['Vul_EW'], r['Vul_EW']][idirection]
                s = score(
                    s-1,
                    CDHSN.index(
                        r[MakeColName(prefix, 'Suit', suit, direction)]),
                    len(r[MakeColName(prefix, 'Double', suit, direction)]),
                    idirection,
                    vul,
                    r[MakeColName(prefix, 'Result', suit, direction)]
                )
            else:
                s = pd.NA
        scoresl.append(s)
    AssignToColumn(df, MakeColName(prefix, 'Score',
                                   suit, direction), scoresl, 'Int16')

    # # change to favored type
    # AssignToColumn(df, MakeColName(prefix, 'Score', suit, direction), pd.to_numeric(
    #     df[MakeColName(prefix, 'Score', suit, direction)], errors='coerce'), 'Int16')
    # df[MakeColName(prefix, 'Bid', suit, direction)] = df[MakeColName(
    #     prefix, 'Bid', suit, direction)].astype('string')  # change to 'string' exension type
    # df[MakeColName(prefix, 'Double', suit, direction)] = df[MakeColName(
    #     prefix, 'Double', suit, direction)].astype('string')  # change to 'string' exension type
    # df[MakeColName(prefix, 'Suit', suit, direction)] = df[MakeColName(
    #     prefix, 'Suit', suit, direction)].astype('string')  # change to 'string' exension type
    # df[MakeColName(prefix, 'Type', suit, direction)] = df[MakeColName(
    #     prefix, 'Type', suit, direction)].astype('string')  # change to 'string' exension type

    return


def highlight_last_max(data, colormax='antiquewhite', colormaxlast='lightgreen', fillna=-1):
    colormax_attr = f'background-color: {colormax}'
    colormaxlast_attr = f'background-color: {colormaxlast}'
    data = data.fillna(fillna)
    if (data == fillna).all():
        return ['']*len(data)
    max_value = data.max()
    is_max = [colormax_attr if v == max_value else '' for v in data]
    is_max[len(data) - list(reversed(data)).index(max_value) -
           1] = colormaxlast_attr
    return is_max


def highlight_last_min(data, colormin='antiquewhite', colorminlast='lightgreen', fillna=-1):
    colormin_attr = f'background-color: {colormin}'
    colorminlast_attr = f'background-color: {colorminlast}'
    data = data.fillna(fillna)
    if (data == fillna).all():
        return ['']*len(data)
    min_value = data.min()
    is_min = [colormin_attr if v == min_value else '' for v in data]
    is_min[len(data) - list(reversed(data)).index(min_value) -
           1] = colorminlast_attr
    return is_min

# scoredf.style.apply(highlight_last_max,axis=1)


def ListOfClubsToProcess(clubNumbers, inputFiles, outputFiles, clubsPath, forceRewriteOfOutputFiles, deleteOutputFiles, sort=True, reverse=True):
    listOfClubs = []
    for clubNumber in clubNumbers:
        clubDir = clubsPath.joinpath(clubNumber.name)
        # all input files must exist
        if sum([not clubDir.joinpath(inputFileToProcess).exists() for inputFileToProcess in inputFiles]) != 0:
            print_to_log_info(
                f'ListOfClubsToProcess: Club {clubNumber.name} has some missing input files: {inputFiles}: skipping.')
            continue
        # creating list of input file sizes, first file only, for later sorting.
        listOfClubs.append((clubDir.joinpath(inputFiles[0]).stat().st_size, clubNumber, clubDir,
                            inputFiles, outputFiles))
        # should output files be removed?
        if not forceRewriteOfOutputFiles and not deleteOutputFiles:
            continue
        # remove existing output files
        for outputFileToProcess in outputFiles:
            ofile = clubDir.joinpath(outputFileToProcess)
            if ofile.exists():
                ofile.unlink()  # remove output files

    # actually doesn't always seem to help performance by ordering files by size. Counter intuitive.
    # order by largest files first for optimization of multiprocessing
    if sort:
        listOfClubs.sort(key=lambda l: l[0], reverse=reverse)
    return listOfClubs


def Categorify(df):

    objectColumns = df.select_dtypes(['object']).columns
    assert len(objectColumns) == 0

    categoryColumns = df.select_dtypes(['category']).columns
    assert len(categoryColumns) == 0

    stringColumns = df.select_dtypes(['string']).columns

    le = preprocessing.LabelEncoder()

    for col in stringColumns:
        #n = len(df[col].unique())
        df[col] = le.fit_transform(df[col])
        #df[col] = le.transform(df[col])

    return list(stringColumns)


def SetupCatCont(df):
    # start by assuming that all numeric columns should be put in continuous column. Todo: revisit this assumption.
    cont_names = df.select_dtypes('number').columns.to_list()
    cat_names = Categorify(df)

    return cat_names, cont_names


def r_mse(pred, y): return round(((((pred-y)**2)**0.5).mean()), 6) # changed from math.sqrt() to **0.5 to eliminate math dependency


def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)


def TranslateSuitSymbol(s):
    return s.replace('♠', 'S').replace('♥', 'H').replace('♦', 'D').replace('♣', 'C')


# todo: implement AVE,PASS,NP,AVE-,AVE+,[0-9]+[FGHL]? Alternatively, throw away as they are inconsequential.
def ComputeMatchPointResults(results):
    numOfPairs = sum([resultsList[1] for resultsList in results])
    sortedResults = sorted(results)
    beat = 0
    scoreToMPs = []
    for r in sortedResults:
        score = r[0]
        count = r[1]
        same = count-1
        mps = beat+same/2
        #scoreToMP = [score, beat, count, mps, mps/(numOfPairs-1)]
        scoreToMP = [score, beat, count, mps, mps if numOfPairs == 1 else mps/(numOfPairs-1)] # numOfPairs == 1 needs testing
        scoreToMPs.append(scoreToMP)
        # print_to_log(scoreToMP)
        beat += count
    assert beat == numOfPairs
    assert len(scoreToMPs) > 0 or numOfPairs == 0
    return numOfPairs, scoreToMPs


# Add a single score to a match point dict. Returns new dict of tuples (score, beats, ties, matchpoints, pct)
def MatchPointScoreUpdate(score,mps):
    sorted_mps = sorted(mps.items(),reverse=True)
    mps = {}
    top = 0
    for k,v in sorted_mps:
        beats = v[1]+v[2]+1
        if top == 0: # first time
            top = beats
        if score > k: # beats and ties unchanged
            if score not in mps: # insert new score
                mps[score] = (score,beats,0,beats,beats/top)  # add low board
            beats = v[1]
            ties = v[2]
        elif score == k: # add a tie
            beats = v[1]
            ties = v[2]+1
        else: # add a beat
            beats = v[1]+1
            ties = v[2]
        beats_ties = beats+ties/2
        pct = round(beats_ties/top,2)
        mps[k] = (k,beats,ties,beats_ties,pct)
    if score not in mps:
        mps[score] = (score,0,0,0.0,0.0)  # insert new low bord
# todo: asserts - check sorted order - check key and score - check sum of beats and ties.
    return mps



def CreateTCGDictEventBoard(eb, cg, d):
    numOfPairs, scoreToMPs = ComputeMatchPointResults(cg)
    d[eb] = [numOfPairs, scoreToMPs]
    for br in scoreToMPs:
        d[eb+'_'+str(br[0])] = br
    return d


def CreateTcgDict(bdf, d):
    for eb, cg in zip(bdf['EventBoard'], bdf['Results']):
        CreateTCGDictEventBoard(eb, cg, d)
    return d


#def CreateTcgDict(bdf, d):
#    for eb, cg in zip(bdf['EventBoard'], bdf['Results']):
#        numOfPairs, scoreToMPs = ComputeMatchPointResults(cg)
#        d[eb] = [numOfPairs, scoreToMPs]
#        for br in scoreToMPs:
#            d[eb+'_'+str(br[0])] = br
#    return d


def ScoreToMP(score, numOfPairs, scoreToMPs):
    # missing score is computed without counting an additional pair
    if len(scoreToMPs) == 0:
        assert numOfPairs == 0
        return [score, 0, 1, 0, np.nan]
    for scoreToMP in scoreToMPs:
        if score == scoreToMP[0]:
            return scoreToMP
        elif score < scoreToMP[0]:
            count = 1
            beat = mps = scoreToMP[1]
            # was (numOfPairs-1) but raise division by zero. Should be ok as we are only estimating MP.
            return [score, beat, count, mps, mps/numOfPairs]
    return [score, numOfPairs, 1, numOfPairs, 1.0]


def GetMissingMP(tcgd, eventBoard, score):
    if eventBoard not in tcgd:
        return [score, 0, 1, 0, np.nan]
    numOfPairs, scoreToMPs = tcgd[eventBoard]
    return ScoreToMP(score, numOfPairs, scoreToMPs)


def GetTcgMP(tcgd, tcgKey):
    if tcgKey in tcgd:
        scorel = tcgd[tcgKey]
    else:
        i = tcgKey.rindex('_')
        eventBoard = tcgKey[:i]
        score = int(tcgKey[i+1:])
        scorel = GetMissingMP(tcgd, eventBoard, score)
    return scorel


def GetTcgMpPercent(tcgd, tcgKey):
    if tcgKey is pd.NA:
        return np.nan
    scorel = GetTcgMP(tcgd, tcgKey)
    pc = scorel[-1]
    if pc is np.nan:  # only happens if no pairs play board
        return pc
    assert pc >= 0.0 and pc <= 1.0
    
    return pc


def GetTcgMPs(tcgd, keyCol):
    return [GetTcgMpPercent(tcgd, tcgKey) for tcgKey in keyCol]


# simple function to walk a json file printing keys and values.
# usage: json_walk_print('main',data_json) where 'main' is becomes the name of the outer table and data_json is a string containing json.

def json_walk_print(key,value):
    if type(value) is dict:
        #print_to_log('dict:'+key)
        for k,v in value.items():
            kk = key+'.'+k
            json_walk_print(kk,v)
    elif type(value) is list:
        #print('list:'+key)
        for n,v in enumerate(value):
            kk = key+'['+str(n)+']'
            json_walk_print(kk,v)
    else:
        if type(value) is str:
            value = '"'+value+'"'
        print_to_log_debug(key+'='+str(value))
    return


# walk a json file building a table suitable for generating SQL statements.
# usage: json_to_sql_walk(tables,'main',data_json,primary_keys) where 'main' is first table and data_json is a string containing json.
def sql_create_tables(tables,key,value):
    #print_to_log(tables, key, value)
    #print_to_log(f"{key}={value}")
    splited = key.split('.')
    tableName = splited[-3]
    fieldId = splited[-2]
    fieldName = splited[-1]
    #print_to_log("ct:", tableName, fieldId, fieldName, type(value))
    # removed assert as they were json schema specific
    #assert not tableName[0].isdigit(), [tableName, fieldId, fieldName, type(value)]
    #assert fieldId[0].isdigit(), [tableName, fieldId, fieldName, type(value)]
    #assert not fieldName[0].isdigit(), [tableName, fieldId, fieldName, type(value)]
    if fieldName in tables[tableName][fieldId]:
        #print_to_log(type(tables[tableName][fieldId][fieldName]))
        #print_to_log(tableName,fieldId,fieldName)
        assert type(tables[tableName][fieldId][fieldName]) is list
        if type(value) is list:
            tables[tableName][fieldId][fieldName] += value
        # set will return unique values from list but all must be same type e.g. str
        elif value not in tables[tableName][fieldId][fieldName]:
            tables[tableName][fieldId][fieldName].append(value)
        # list must consist of only unique values. award pigment issue
        # careful: set is non-deterministic so values in a list(set(l)) can become reordered!
        assert len(set(tables[tableName][fieldId][fieldName])) == len(tables[tableName][fieldId][fieldName])
    else:
        tables[tableName][fieldId][fieldName] = value
    return


def json_to_sql_walk(tables,key,last_id,uid,value,primary_keys):
    #print_to_log(tables,key,last_id,uid,value)
    if type(value) is dict:
        #print_to_log('dict:',key,uid)
        if any([pk in value for pk in primary_keys]):
            for pk in primary_keys:
                if pk in value:
                    last_id = key.split('.')[-1]
                    uid = [str(value[pk])]
                    if key.count('.') > 0:
                        sql_create_tables(tables,key,'-'.join(uid))
        elif all(not k.isdigit() for k in value.keys()):
            sql_create_tables(tables,key+'.'+'-'.join(uid)+'.id','-'.join(uid)) # create PRIMARY KEY column of 'id'
            sql_create_tables(tables,key+'.'+'-'.join(uid)+'.'+last_id,uid[0]) # create parent column using last_id and uid[0] (first id)
            if key.count('.') > 0:
                sql_create_tables(tables,key,['-'.join(uid)])
        for k,v in value.items():
            if all(kk.isdigit() for kk in value.keys()):
                json_to_sql_walk(tables,key,last_id,uid+[k],v,primary_keys)
            else:
                json_to_sql_walk(tables,key+'.'+'-'.join(uid)+'.'+k,last_id,uid,v,primary_keys)
    elif type(value) is list:
        #print_to_log('list:',key,uid)
        if len(value) > 0: # turn empty lists into NULL?
            #print_to_log("empty list:",key)
            sql_create_tables(tables,key,[])
        for n,v in enumerate(value):
            json_to_sql_walk(tables,key,last_id,uid+[str(n)],v,primary_keys)
    else:
        sql_create_tables(tables,key,value)
    return


# Create a file of SQL INSERT commands from table
def CreateSqlFile(tables,f,primary_keys):
    print("PRAGMA foreign_keys = OFF;", file=f) # is this still necessary????
    
    for k,v in tables.items():
        assert type(v) is defaultdict
        #print(f"DELETE FROM [{k}];", file=f) # delete all rows
        for kk,vv in v.items():
            assert type(vv) is dict
            s = '\",\"'.join(vvv for vvv in vv.keys()) # backslashes can't be included within format {}
            print(f"INSERT INTO \"{k}\" (\"{s}\")", file=f)
            values = []
            for kkk,vvv in vv.items():
                #print_to_log(kkk,vvv)
                if type(vvv) is str:
                    values.append('\''+vvv.replace('\'','\'\'')+'\'') # escape embedded double-quotes with sql's double double-quotes
                elif vvv is None:
                    values.append("NULL")
                elif type(vvv) is list:
                    #print_to_log("list:",kkk,vvv)
                    # patch - 2023-01-19 - added - added code to quote list items if they're strings
                    if len(vvv)>0 and isinstance(vvv[0],str):
                        values.append('\'["'+'","'.join(str(vvvv).replace('\'','\'\'') for vvvv in vvv)+'"]\'') # escape embedded double-quotes with sql's double double-quotes
                    else:
                        values.append('\'['+','.join(str(vvvv).replace('\'','\'\'') for vvvv in vvv)+']\'') # escape embedded double-quotes with sql's double double-quotes
                else:
                    values.append(vvv)
            print(f"VALUES({','.join(str(vvvv) for vvvv in values)})", file=f)
            # DO UPDATE SET updated_at=excluded.updated_at
            s = ','.join('\"'+vvv+'\"=excluded.\"'+vvv+'\"' for vvv in vv.keys()) # backslashes can't be included within format {}
            assert any([pk in vv for pk in primary_keys]),[primary_keys,vv]
            for pk in primary_keys:
                if pk in vv:
                    print(f"ON CONFLICT({pk}) DO UPDATE SET {s}", file=f, end='')
            #print_to_log('created_at' in vv.keys(),'updated_at' in vv.keys())
            assert ('created_at' in vv.keys()) == ('updated_at' in vv.keys())
            if 'created_at' in vv.keys():
                print(f"\nWHERE excluded.\"updated_at\" > \"updated_at\" OR (excluded.\"updated_at\" = \"updated_at\" AND excluded.\"created_at\" > \"created_at\")", file=f, end='')
            print(";\n",file=f)
    return


# Automatically create sql tables file.
# Will require further editing of most fields: fix PRIMARY KEY, make NOT NULL, change VARTYPE to INT, REAL, remove trailing comma.
def CreateSqlTablesFile(f,tables,primary_keys):
    assert type(tables) is defaultdict
    print(f'PRAGMA journal_mode=WAL;', file=f)
    print(file=f)
    for k,v in tables.items():
        print(f'DROP TABLE IF EXISTS "{k}";', file=f)
        print(file=f)
        print(f'CREATE TABLE "{k}" (', file=f)
        assert type(v) is defaultdict
        for kk,vv in v.items():
            #print_to_log('uid:','.'.join([k,kk]))
            assert kk[0].isdigit()
            assert type(vv) is dict
            for kkk,vvv in vv.items():
                #print_to_log('3:','.'.join([k,kkk]))
                assert type(vvv) is not list or type(vvv) is not dict
                if kkk in primary_keys:
                    print(f'"{kkk}" INT NOT NULL PRIMARY KEY,', file=f)
                else:
                    print(f'"{kkk}" VARCHAR NULL,', file=f) # or NOT
        for kkk,vvv in vv.items():
            #print_to_log(kkk,tables.keys())
            if kkk in tables:
                print(f'-- list of VARCHAR', file=f)
                print(f'FOREIGN KEY ("{kkk}") REFERENCES "{kkk}"(id) ON DELETE NO ACTION,', file=f)
        print(');', file=f)
        print(file=f)
# only create file if it doesn't exist. Must manually delete file if new version is wanted.
#create_tables_sql_file = pathlib.Path("acbl_tournament_sessions_schema.sql")
#if not create_tables_sql_file.exists():
#    with open(create_tables_sql_file,'w',encoding='utf-8') as f:
#        CreateSqlTablesFile(f,tables)
#tables = defaultdict(lambda :defaultdict(dict))
#json_to_sql_walk(tables,'events',"",[],data_json,primary_keys)
    

# initializations
scoresd, setScoresd, makeScoresd = ScoreDicts()
