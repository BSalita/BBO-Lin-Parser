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

import pandas as pd
import polars as pl
from pprint import pprint # obsolete?
import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd().parent.parent.joinpath('mlBridgeLib'))) # removed .parent
sys.path.append(str(pathlib.Path.cwd().parent.parent.joinpath('acbllib'))) # removed .parent
sys.path
import mlBridgeLib
import acbllib


# obsolete?
#from tenacity import retry, wait_random_exponential, stop_after_attempt
#from termcolor import colored
#import dotenv
#from dotenv import load_dotenv
#import os
#import inspect
# def pretty_print_conversation(messages):
#     role_to_color = {
#         "system": "red",
#         "user": "green",
#         "assistant": "blue",
#         "function": "magenta",
#     }
#     formatted_messages = []
#     for message in messages:
#         if message["role"] == "system":
#             formatted_messages.append(f"system: {message['content']}\n")
#         elif message["role"] == "user":
#             formatted_messages.append(f"user: {message['content']}\n")
#         elif message["role"] == "assistant":
#             if message.get("function_call"):
#                 formatted_messages.append(
#                     f"assistant: {message['function_call']}\n")
#             else:
#                 formatted_messages.append(f"assistant: {message['content']}\n")
#         elif message["role"] == "function":
#             formatted_messages.append(f"assistant: {message['content']}\n")
#             #formatted_messages.append(f"function ({message['name']}): {message['content']}\n")
#     for formatted_message in formatted_messages:
#         print_to_log(
#             colored(
#                 formatted_message,
#                 role_to_color[messages[formatted_messages.index(
#                     formatted_message)]["role"]],
#             )
#         )


# merge acbl json dicts into logically related dicts. dicts will be used to create dfs
def json_dict_to_df(d,kl,jdl):
    print_to_log_debug(kl)
    dd = {}
    d[kl] = dd
    assert not isinstance(jdl,dict)
    for i,jd in enumerate(jdl):
        for k,v in jd.items():
            kkl = kl+(k,i)
            print_to_log_debug(i,kl,k,kkl)
            #time.sleep(.00001)
            if isinstance(v,list):
                print_to_log_debug('\n',type(v),kkl,v)
                json_dict_to_df(d,kkl,v)
                print_to_log_debug('list:',kkl,len(d[kkl]))
            elif isinstance(v,dict):
                #kkl = kl+(k,)
                print_to_log_debug('\n',type(v),kkl,v)
                json_dict_to_df(d,kkl,[v])
                print_to_log_debug('dict:',kkl,len(d[kkl]))
            else:
                if k not in dd:
                    dd[k] = []
                dd[k].append(v)
            #assert k != 'points',[kl,k,type(v),v]
    return d


# todo: obsolete?
# todo: if dtype isnumeric() downcast to minimal size. Some columns may have dtype of int64 because of sql declaration ('Board').
def convert_to_best_dtypex(k,v):
    vv = v.convert_dtypes(infer_objects=True)
    vvv = vv.copy()
    for col in vv.columns:
        print_to_log_debug(col,vvv[col].dtype)
        # todo: special cases. maybe should be done before writing to acbl_club_results.sqlite?
        if col in ['ns_score','ew_score']:
            vvv[col] = vvv[col].replace('PASS','0')
        elif col == 'result':
            vvv[col] = vvv[col].replace('+','0').replace('=','0').replace('','0') # don't use .str. and all 3 are needed.
        elif col == 'tricks_taken':
            vvv[col] = vvv[col].replace('','0')
        if vvv[col].dtype == 'string' and vvv[col].notna().all() and vvv[col].ne('').all():
            print_to_log_debug(f"String: {col}")
            try:
                if vvv[col].str.contains('.',regex=False).any():
                    print_to_log_debug(f"Trying to convert {col} to float")
                    converted_values = pd.to_numeric(vvv[col], downcast='float', errors='raise')
                elif vvv[col].str.contains('-',regex=False).any():
                    print_to_log_debug(f"Trying to convert {col} to integer")
                    converted_values = pd.to_numeric(vvv[col], downcast='integer', errors='raise')
                else:
                    print_to_log_debug(f"Trying to convert {col} to unsigned")
                    converted_values = pd.to_numeric(vvv[col], downcast='unsigned', errors='raise')
                vvv[col] = converted_values
                print_to_log_debug(f"Converted {col} to {vvv[col].dtype}")
            except ValueError:
                print_to_log(logging.WARNING, f"Can't convert {col} to float. Keeping as string")
    print_to_log_debug(f"dfs_dtype_conversions['{k}'] = "+'{')
    for col in vvv.columns:
        print_to_log_debug(f"    '{col}':'{v[col].dtype},{vv[col].dtype},{vvv[col].dtype}',")
    print_to_log_debug("}\n")
    return vvv


# g_all_functions_in_module = {n:f for n,f in inspect.getmembers(sys.modules[__name__], inspect.isfunction)}

def json_dict_to_types(json_dict,root_name,path):
    dfs = {}
    root = []
    df = pd.json_normalize(json_dict,path,max_level=0)
    for k,v in df.items():
        if isinstance(v,dict):
            assert k not in dfs, k
            dfs[k] = pd.DataFrame(v)
            if all(isinstance(kk,int) or (isinstance(kk,str) and kk.isnumeric()) for kk,vv in v.items()): # dict but with list like indices
                dfs[k] = dfs[k].T
        elif isinstance(v,list):
            assert k not in dfs, k
            dfs[k] = pd.DataFrame(v)
        else:
            root.append({k:v})
    assert k not in dfs, k
    dfs[root_name] = pd.DataFrame(root)
    return dfs


def create_club_dfs(acbl_number,event_url):
    data = acbllib.get_club_results_details_data(event_url)
    if data is None:
        return None
    dfs = {}
    dfs['event'] = pd.json_normalize(data,max_level=0)
    for k,v in dfs['event'].items():
        if isinstance(v[0],dict) or isinstance(v[0],list):
            assert k not in dfs, k
            df = pd.json_normalize(data,max_level=0)[k]
            # must test whether df is all scalers. Very difficult to debug.
            if isinstance(v[0],dict) and not any([isinstance(vv,dict) or isinstance(vv,list) for kk,vv in df[0].items()]):
                dfs[k] = pd.DataFrame.from_records(df[0],index=[0]) # must use from_records to avoid 'all values are scaler must specify index' error
            else:
                dfs[k] = pd.DataFrame.from_records(df[0])
            dfs['event'].drop(columns=[k],inplace=True)
            #if all(isinstance(kk,int) or (isinstance(kk,str) and kk.isnumeric()) for kk,vv in v.items()):
    dfs['hand_records'] = pd.json_normalize(data,['sessions','hand_records'])
    dfs['strat_place'] = pd.json_normalize(data,['sessions','sections','pair_summaries','strat_place'])
    dfs['sections'] = pd.json_normalize(data,['sessions','sections'])
    dfs['boards'] = pd.json_normalize(data,['sessions','sections','boards'])
    dfs['pair_summaries'] = pd.json_normalize(data,['sessions','sections','pair_summaries'])
    dfs['players'] = pd.json_normalize(data,['sessions','sections','pair_summaries','players'])
    dfs['board_results'] = pd.json_normalize(data,['sessions','sections','boards','board_results'])
    return dfs


def merge_clean_augment_tournament_dfs(dfs, dfs_results, acbl_api_key, acbl_number):

    print_to_log_info('dfs keys:',dfs.keys())

    df = pd.DataFrame({k:[v] for k,v in dfs.items() if not (isinstance(v,dict) or isinstance(v,list))})
    print_to_log_info('df:\n', df)
    assert len(df) == 1, len(df)
    
    print_to_log_info('dfs session:',type(dfs['session']))
    df_session = pd.DataFrame({k:[v] for k,v in dfs['session'].items() if not (isinstance(v,dict) or isinstance(v,list))})
    assert len(df_session) == 1, len(df_session)
    print_to_log_info({k:pd.DataFrame(v) for k,v in dfs['session'].items() if (isinstance(v,dict) or isinstance(v,list))})

    print_to_log_info('dfs event:',type(dfs['event']))
    df_event = pd.DataFrame({k:[v] for k,v in dfs['event'].items() if not (isinstance(v,dict) or isinstance(v,list))})
    assert len(df_event) == 1, len(df_event)
    print_to_log_info({k:pd.DataFrame(v) for k,v in dfs['event'].items() if (isinstance(v,dict) or isinstance(v,list))})

    print_to_log_info('dfs tournament:',type(dfs['tournament']))
    df_tournament = pd.DataFrame({k:[v] for k,v in dfs['tournament'].items() if not (isinstance(v,dict) or isinstance(v,list))})
    assert len(df_tournament) == 1, len(df_tournament)
    print_to_log_info({k:pd.DataFrame(v) for k,v in dfs['tournament'].items() if (isinstance(v,dict) or isinstance(v,list))})

    for col in df.columns:
        print_to_log_debug('cols:',col,df[col].dtype)

    # dfs scalers: ['_id', '_event_id', 'id', 'session_number', 'start_date', 'start_time', 'description', 'sess_type', 'box_number', 'is_online', 'results_available', 'was_not_played', 'results_last_updated']
    # dfs dicts: ['tournament', 'event', 'handrecord', 'sections']
    # dfs lists: ['overalls']
   
    print_to_log_info('dfs_results tournament:',type(dfs_results['tournament']))
    df_results_tournament = pd.DataFrame({k:[v] for k,v in dfs_results['tournament'].items() if not (isinstance(v,dict) or isinstance(v,list))})
    assert len(df_results_tournament) == 1, len(df_results_tournament)
    print_to_log_info({k:pd.DataFrame(v) for k,v in dfs_results['tournament'].items() if (isinstance(v,dict) or isinstance(v,list))})

    print_to_log_info('dfs_results event:',type(dfs_results['event']))
    df_results_event = pd.DataFrame({k:[v] for k,v in dfs_results['event'].items() if not (isinstance(v,dict) or isinstance(v,list))})
    assert len(df_event) == 1, len(df_event)
    print_to_log_info({k:pd.DataFrame(v) for k,v in dfs_results['event'].items() if (isinstance(v,dict) or isinstance(v,list))})

    print_to_log_info('dfs_results overalls:',type(dfs_results['overalls']))
    df_results_overalls = pd.DataFrame(dfs_results['overalls'])
    #assert len(df_results_overalls) == 1, len(df_results_overalls)
    print_to_log_info(pd.DataFrame(dfs_results['overalls']))

    print_to_log_info('dfs_results handrecord:',type(dfs_results['handrecord']))
    df_results_handrecord = pd.DataFrame(dfs_results['handrecord'])
    #assert len(df_results_handrecord) == 1, len(df_results_handrecord)
    print_to_log_info(pd.DataFrame(dfs_results['handrecord']))

    print_to_log_info('dfs_results sections:',type(dfs_results['sections']))
    df_results_sections = pd.DataFrame(dfs_results['sections'])

    df_board_results = pd.DataFrame()
    for i,section in df_results_sections.iterrows():
        br = pd.DataFrame(section['board_results'])
        # todo: what to do with sections not containing acbl_number? concat all sections? concat may be correct since they may be included in matchpoint calculations.
        if all(br['pair_acbl'].map(lambda x: int(acbl_number) not in x)): # if acbl_number is not in this section then skip.(?)
            continue
        df_board_results = pd.concat([df_board_results,br],axis='rows')
        ns_df = df_board_results[df_board_results['orientation'].eq('N-S')]
        ew_df = df_board_results[df_board_results['orientation'].eq('E-W')][['board_number','pair_number','pair_names','pair_acbl','score','match_points','percentage']]
        df_board_results = pd.merge(ns_df,ew_df,left_on=['board_number','opponent_pair_number'],right_on=['board_number','pair_number'],suffixes=('_ns','_ew'),how='left')
        df_board_results.drop(['opponent_pair_number','opponent_pair_names'],inplace=True,axis='columns')
        df_board_results.rename({
            'board_number':'Board',
            'contract':'Contract',
            'score_ns':'Score_NS',
            'score_ew':'Score_EW',
            'match_points_ns':'MatchPoints_NS',
            'match_points_ew':'MatchPoints_EW',
            'percentage_ns':'Pct_NS',
            'percentage_ew':'Pct_EW',
            'pair_number_ns':'Pair_Number_NS',
            'pair_number_ew':'Pair_Number_EW',
            'session_number':'Session',
        },axis='columns',inplace=True)
        df_board_results['pair_direction'] = df_board_results['orientation'].map({'N-S':'NS','E-W':'EW'})
        df_board_results['player_number_n'] = df_board_results.apply(lambda r: r['pair_acbl_ns'][0],axis='columns').astype('string')
        df_board_results['player_number_s'] = df_board_results.apply(lambda r: r['pair_acbl_ns'][1],axis='columns').astype('string')
        df_board_results['player_number_e'] = df_board_results.apply(lambda r: r['pair_acbl_ew'][0],axis='columns').astype('string')
        df_board_results['player_number_w'] = df_board_results.apply(lambda r: r['pair_acbl_ew'][1],axis='columns').astype('string')
        df_board_results['player_name_n'] = df_board_results.apply(lambda r: r['pair_names_ns'][0],axis='columns')
        df_board_results['player_name_s'] = df_board_results.apply(lambda r: r['pair_names_ns'][1],axis='columns')
        df_board_results['player_name_e'] = df_board_results.apply(lambda r: r['pair_names_ew'][0],axis='columns')
        df_board_results['player_name_w'] = df_board_results.apply(lambda r: r['pair_names_ew'][1],axis='columns')
        # todo: get from masterpoint dict
        #df_board_results['Club'] = '12345678' # why is this needed?
        df_board_results['Session'] = section['session_id']
        df_board_results['mp_total_n'] = 300
        df_board_results['mp_total_e'] = 300
        df_board_results['mp_total_s'] = 300
        df_board_results['mp_total_w'] = 300
        df_board_results['MP_Sum_NS'] = 300+300
        df_board_results['MP_Sum_EW'] = 300+300
        df_board_results['MP_Geo_NS'] = 300*300
        df_board_results['MP_Geo_EW'] = 300*300
        df_board_results['declarer'] = df_board_results['declarer'].map(lambda x: x[0].upper() if len(x) else None) # None is needed for PASS
        df_board_results['Pct_NS'] = df_board_results['Pct_NS'].div(100)
        df_board_results['Pct_EW'] = df_board_results['Pct_EW'].div(100)
        df_board_results['Table'] = None # todo: is this right?
        df_board_results['Round'] = None # todo: is this right?
        df_board_results['tb_count'] = None # todo: is this right?
        df_board_results['dealer'] = df_board_results['Board'].map(mlBridgeLib.BoardNumberToDealer)
        df_board_results['iVul'] = df_board_results['Board'].map(mlBridgeLib.BoardNumberToVul).astype('uint8') # 0 to 3
        df_board_results['event_id'] = section['session_id'].astype('int32') # for club compatibility
        df_board_results['section_name'] = section['section_label'] # for club compatibility
        df_board_results['section_id'] = df_board_results['event_id']+'-'+df_board_results['section_name'] # for club compatibility
        df_board_results['Date'] = pd.to_datetime(df_event['start_date'][0]) # converting to datetime64[ns] for human readable display purposes but will create 'iDate' (int64) in augment
        df_board_results['game_type'] = df_event['game_type'].astype('category') # for club compatibility
        df_board_results['event_type'] = df_event['event_type'].astype('category') # for club compatibility
        df_board_results['mp_limit'] = df_event['mp_limit'].astype('category') # for club compatibility
        df_board_results['mp_color'] = df_event['mp_color'].astype('category') # for club compatibility
        df_board_results['mp_rating'] = df_event['mp_rating'].astype('category') # for club compatibility
        board_to_brs_d = dict(zip(df_results_handrecord['board_number'],mlBridgeLib.hrs_to_brss(df_results_handrecord)))
        df_board_results['board_record_string'] = df_board_results['Board'].map(board_to_brs_d)
        df_board_results.drop(['orientation','pair_acbl_ns', 'pair_acbl_ew', 'pair_names_ns', 'pair_names_ew'],inplace=True,axis='columns')


    df = clean_validate_df(df_board_results)
    df, sd_cache_d, matchpoint_ns_d = augment_df(df,{})

    return df, sd_cache_d, matchpoint_ns_d


# obsolete?
def clean_validate_tournament_df(df):

    # par, hand_record_id, DD, Vul, Hands, board_record_string?, ns_score, ew_score, Final_Stand_NS|EW, MatchPoints_NS, MatchPoints_EW, player_number_[nesw], contract, BidLvl, BidSuit, Dbl, Declarer_Direction

    # change clean_validate_club_df to handle these missing columns; par, Pair_Number_NS|EW, table_number, round_number, double_dummy_ns|ew, board_record_string, hand_record_id.

    return df


# obsolete?
def augment_tournament_df(df,sd_cache_d):
    return df, sd_cache_d, {}


def merge_clean_augment_club_dfs(dfs,sd_cache_d,acbl_number): # todo: acbl_number obsolete?

    print_to_log_info('merge_clean_augment_club_dfs: dfs keys:',dfs.keys())

    df_brs = dfs['board_results']
    print_to_log_info(df_brs.head(1))
    assert len(df_brs.filter(regex=r'_[xy]$').columns) == 0,df_brs.filter(regex=r'_[xy]$').columns

    df_b = dfs['boards'].rename({'id':'board_id'},axis='columns')[['board_id','section_id','board_number']]
    print_to_log_info(df_b.head(1))
    assert len(df_b.filter(regex=r'_[xy]$').columns) == 0,df_b.filter(regex=r'_[xy]$').columns

    df_br_b = pd.merge(df_brs,df_b,on='board_id',how='left')
    print_to_log_info(df_br_b.head(1))
    assert len(df_br_b) == len(df_brs)
    assert len(df_br_b.filter(regex=r'_[xy]$').columns) == 0,df_br_b.filter(regex=r'_[xy]$').columns


    df_sections = dfs['sections'].rename({'id':'section_id','name':'section_name'},axis='columns').drop(['created_at','updated_at','transaction_date','pair_summaries','boards'],axis='columns') # ['pair_summaries','boards'] are unwanted dicts
    print_to_log_info(df_sections.head(1))


    df_br_b_sections = pd.merge(df_br_b,df_sections,on='section_id',how='left')
    print_to_log_info(df_br_b_sections.head(1))
    assert len(df_br_b_sections) == len(df_br_b)
    assert len(df_br_b_sections.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections.filter(regex=r'_[xy]$').columns


    df_sessions = dfs['sessions'].rename({'id':'session_id','number':'session_number'},axis='columns').drop(['created_at','updated_at','transaction_date','hand_records','sections'],axis='columns') # ['hand_records','sections'] are unwanted dicts
    # can't convert to int64 because SHUFFLE is a valid hand_record_id. Need to treat as string.
    # df_sessions['hand_record_id'] = df_sessions['hand_record_id'].astype('int64') # change now for merge
    print_to_log_info(df_sessions.head(1))


    df_br_b_sections_sessions = pd.merge(df_br_b_sections,df_sessions,on='session_id',how='left')
    print_to_log_info(df_br_b_sections_sessions.head(1))
    assert len(df_br_b_sections_sessions) == len(df_br_b_sections)
    assert len(df_br_b_sections_sessions.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections_sessions.filter(regex=r'_[xy]$').columns # to fix, drop duplicated column names


    df_clubs = dfs['club'].rename({'id':'event_id','name':'club_name','type':'club_type'},axis='columns').drop(['created_at','updated_at','transaction_date'],axis='columns') # name and type are renamed to avoid conflict with df_events
    print_to_log_info(df_clubs.head(1))


    df_br_b_sections_sessions_clubs = pd.merge(df_br_b_sections_sessions,df_clubs,on='event_id',how='left')
    print_to_log_info(df_br_b_sections_sessions_clubs.head(1))
    assert len(df_br_b_sections_sessions_clubs) == len(df_br_b_sections)
    assert len(df_sections.filter(regex=r'_[xy]$').columns) == 0,df_sections.filter(regex=r'_[xy]$').columns

        
    df_events = dfs['event'].rename({'id':'event_id','club_name':'event_club_name','type':'event_type'},axis='columns').drop(['created_at','updated_at','transaction_date','deleted_at'],axis='columns')
    print_to_log_info(df_events.head(1))


    df_br_b_sections_sessions_events = pd.merge(df_br_b_sections_sessions_clubs,df_events,on='event_id',how='left')
    print_to_log_info(df_br_b_sections_sessions_events.head(1))
    assert len(df_br_b_sections_sessions_events) == len(df_br_b_sections_sessions_clubs)
    assert len(df_br_b_sections_sessions_events.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections_sessions_events.filter(regex=r'_[xy]$').columns


    df_pair_summaries = dfs['pair_summaries'].rename({'id':'pair_summary_id'},axis='columns').drop(['created_at','updated_at','transaction_date'],axis='columns')
    print_to_log_info(df_pair_summaries.head(1))

    # todo: merge df_pair_summaries with strat_place. issue is that strat_place has multiple rows per pair_summary_id
    df_pair_summaries_strat = df_pair_summaries
    # df_strat_place = dfs['strat_place'].rename({'rank':'strat_rank','type':'strat_type'},axis='columns').drop(['id','created_at','updated_at','transaction_date'],axis='columns')
    # print_to_log(df_strat_place.head(1))

    # df_pair_summaries_strat = pd.merge(df_pair_summaries,df_strat_place,on='pair_summary_id',how='left')
    # print_to_log(df_pair_summaries_strat.head(1))
    # assert len(df_pair_summaries_strat.filter(regex=r'_[xy]$').columns) == 0,df_pair_summaries_strat.filter(regex=r'_[xy]$').columns

    df_br_b_pair_summary_ns = df_pair_summaries_strat[df_pair_summaries_strat['direction'].eq('NS')].add_suffix('_ns').rename({'pair_number_ns':'ns_pair','section_id_ns':'section_id'},axis='columns')
    assert len(df_br_b_pair_summary_ns.filter(regex=r'_[xy]$').columns) == 0,df_br_b_pair_summary_ns.filter(regex=r'_[xy]$').columns
    df_br_b_pair_summary_ew = df_pair_summaries_strat[df_pair_summaries_strat['direction'].eq('EW')].add_suffix('_ew').rename({'pair_number_ew':'ew_pair','section_id_ew':'section_id'},axis='columns')
    assert len(df_br_b_pair_summary_ew.filter(regex=r'_[xy]$').columns) == 0,df_br_b_pair_summary_ew.filter(regex=r'_[xy]$').columns

    df_players = dfs['players'].drop(['id','created_at','updated_at','transaction_date'],axis='columns').rename({'id_number':'player_number','name':'player_name'},axis='columns')
    player_n = df_players.groupby('pair_summary_id').first().reset_index().add_suffix('_n').rename({'pair_summary_id_n':'pair_summary_id_ns'},axis='columns')
    player_s = df_players.groupby('pair_summary_id').last().reset_index().add_suffix('_s').rename({'pair_summary_id_s':'pair_summary_id_ns'},axis='columns')
    player_e = df_players.groupby('pair_summary_id').first().reset_index().add_suffix('_e').rename({'pair_summary_id_e':'pair_summary_id_ew'},axis='columns')
    player_w = df_players.groupby('pair_summary_id').last().reset_index().add_suffix('_w').rename({'pair_summary_id_w':'pair_summary_id_ew'},axis='columns')

    player_ns = pd.merge(player_n,player_s,on='pair_summary_id_ns',how='left')
    print_to_log_info(player_ns.head(1))
    assert len(player_ns) == len(player_n)
    assert len(player_ns.filter(regex=r'_[xy]$').columns) == 0,player_ns.filter(regex=r'_[xy]$').columns
    player_ew = pd.merge(player_e,player_w,on='pair_summary_id_ew',how='left')
    print_to_log_info(player_ew.head(1))
    assert len(player_ew) == len(player_e)
    assert len(player_ew.filter(regex=r'_[xy]$').columns) == 0,player_ew.filter(regex=r'_[xy]$').columns

    # due to an oddity with merge(), must never merge on a column that has NaNs. This section avoids that but at the cost of added complexity.
    df_pair_summary_players_ns = pd.merge(df_br_b_pair_summary_ns,player_ns,on='pair_summary_id_ns',how='left')
    assert len(df_pair_summary_players_ns) == len(df_br_b_pair_summary_ns)
    df_pair_summary_players_ew = pd.merge(df_br_b_pair_summary_ew,player_ew,on='pair_summary_id_ew',how='left')
    assert len(df_pair_summary_players_ew) == len(df_br_b_pair_summary_ew)
    #df_pair_summary_players = pd.merge(df_pair_summary_players_ns,df_pair_summary_players_ew,how='left') # yes, on is not needed
    #assert len(df_pair_summary_players) == len(df_pair_summary_players_ns) # likely this is an issue on an EW sitout. Need to compare ns,ew lengths and how on the longer one?
    df_br_b_sections_sessions_events_pair_summary_players = pd.merge(df_br_b_sections_sessions_events,df_pair_summary_players_ns,on=('section_id','ns_pair'),how='left') # yes, requires inner. Otherwise right df non-on columns will be NaNs.
    print_to_log_info(df_br_b_sections_sessions_events_pair_summary_players.head(1))
    assert len(df_br_b_sections_sessions_events_pair_summary_players) == len(df_br_b_sections_sessions_events)
    assert len(df_br_b_sections_sessions_events_pair_summary_players.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections_sessions_events_pair_summary_players.filter(regex=r'_[xy]$').columns
    df_br_b_sections_sessions_events_pair_summary_players = pd.merge(df_br_b_sections_sessions_events_pair_summary_players,df_pair_summary_players_ew,on=('section_id','ew_pair'),how='left') # yes, requires inner. Otherwise right df non-on columns will be NaNs.
    print_to_log_info(df_br_b_sections_sessions_events_pair_summary_players.head(1))
    assert len(df_br_b_sections_sessions_events_pair_summary_players) == len(df_br_b_sections_sessions_events)
    assert len(df_br_b_sections_sessions_events_pair_summary_players.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections_sessions_events_pair_summary_players.filter(regex=r'_[xy]$').columns

    df_hrs = dfs['hand_records'].rename({'hand_record_set_id':'hand_record_id'},axis='columns').drop(['points.N','points.E','points.S','points.W'],axis='columns') # don't want points (HCP) from hand_records. will compute later.
    print_to_log_info(df_hrs.head(1))

    df_br_b_sections_sessions_events_pair_summary_players_hrs = pd.merge(df_br_b_sections_sessions_events_pair_summary_players,df_hrs.astype({'hand_record_id':'string'}).drop(['id','created_at','updated_at'],axis='columns'),left_on=('hand_record_id','board_number'),right_on=('hand_record_id','board'),how='left')
    print_to_log_info(df_br_b_sections_sessions_events_pair_summary_players_hrs.head(1))
    assert len(df_br_b_sections_sessions_events_pair_summary_players_hrs) == len(df_br_b_sections_sessions_events_pair_summary_players)
    assert len(df_br_b_sections_sessions_events_pair_summary_players_hrs.filter(regex=r'_[xy]$').columns) == 0,df_br_b_sections_sessions_events_pair_summary_players_hrs.filter(regex=r'_[xy]$').columns


    df = df_br_b_sections_sessions_events_pair_summary_players_hrs
    for col in df.columns:
        print_to_log_info('cols:',col,df[col].dtype)

    df.drop(['id','created_at','updated_at','board_id','double_dummy_ns','double_dummy_ew'],axis='columns',inplace=True)

    df.rename({
        'board':'Board',
        'club_id_number':'Club',
        'contract':'Contract',
        'game_date':'Date',
        'ns_match_points':'MatchPoints_NS',
        'ew_match_points':'MatchPoints_EW',
        'ns_pair':'Pair_Number_NS',
        'ew_pair':'Pair_Number_EW',
        'percentage_ns':'Final_Standing_NS',
        'percentage_ew':'Final_Standing_EW',
        'result':'Result',
        'round_number':'Round',
        'ns_score':'Score_NS',
        'ew_score':'Score_EW',
        'session_number':'Session',
        'table_number':'Table',
        'tricks_taken':'Tricks',
       },axis='columns',inplace=True)

    # columns unique to club results
    df = df.astype({
        'board_record_string':'string',
        'Date':'datetime64[ns]', # human-readable date for display. also will create 'iDate' (int64) in augment
        'Final_Standing_NS':'float32',
        'Final_Standing_EW':'float32',
        'hand_record_id':'int64',
        'Pair_Number_NS':'uint16',
        'Pair_Number_EW':'uint16',
        })

    df = clean_validate_df(df)
    df, sd_cache_d, matchpoint_ns_d = augment_df(df,sd_cache_d) # takes 5s

    return df, sd_cache_d, matchpoint_ns_d


def clean_validate_df(df):

    df.rename({'declarer':'Declarer_Direction'},axis='columns',inplace=True)

    # Cleanup all sorts of columns. Create new columns where missing.
    df.drop(df[df['Board'].isna()].index,inplace=True) # https://my.acbl.org/club-results/details/952514
    df['Board'] = df['Board'].astype('uint8')
    assert df['Board'].ge(1).all()

    assert 'Board_Top' not in df.columns
    tops = {}
    for b in df['Board'].unique():
        tops[b] = df[df['Board'].eq(b)]['MatchPoints_NS'].count()-1
        assert tops[b] == df[df['Board'].eq(b)]['MatchPoints_EW'].count()-1
    # if any rows were dropped, the calculation of board's top/pct will be wrong (outside of (0,1)). Need to calculate Board_Top before dropping any rows.
    # PerformanceWarning: DataFrame is highly fragmented.
    df['Board_Top'] = df['Board'].map(tops)
    if set(['Pct_NS', 'Pct_EW']).isdisjoint(df.columns): # disjoint means no elements of set are in df.columns
        # PerformanceWarning: DataFrame is highly fragmented.
        df['Pct_NS'] = df['MatchPoints_NS'].astype('float32').div(df['Board_Top'])
        # PerformanceWarning: DataFrame is highly fragmented.
        df['Pct_EW'] = df['MatchPoints_EW'].astype('float32').div(df['Board_Top'])
    assert set(['Pct_NS', 'Pct_EW', 'Board_Top']).issubset(df.columns) # subset means all elements of the set are in df.columns
    df.loc[df['Pct_NS']>1,'Pct_NS'] = 1 # assuming this can only happen if director adjusts score. todo: print >1 cases.
    assert df['Pct_NS'].between(0,1).all(), [df[~df['Pct_NS'].between(0,1)][['Board','MatchPoints_NS','Board_Top','Pct_NS']]]
    df.loc[df['Pct_EW']>1,'Pct_EW'] = 1 # assuming this can only happen if director adjusts score. todo: print >1 cases.
    assert df['Pct_EW'].between(0,1).all(), [df[~df['Pct_EW'].between(0,1)][['Board','MatchPoints_EW','Board_Top','Pct_EW']]]

    # transpose pair_name (last name, first_name).
    for d in 'NESW':
        df.rename({'player_number_'+d.lower():'Player_Number_'+d},axis='columns',inplace=True)
        # PerformanceWarning: DataFrame is highly fragmented.
        df['iPlayer_Number_'+d] = pd.to_numeric(df['Player_Number_'+d], errors='coerce').fillna(0).astype('int32') # Convert to numeric. Make NaN into 0. Create iPlayer_Number column to match ai model column name. ai likes numerics, hates strings.
        # PerformanceWarning: DataFrame is highly fragmented.
        df['Player_Name_'+d] = df['player_name_'+d.lower()].str.split(',').str[::-1].str.join(' ') # github Copilot wrote this line!
        df.drop(['player_name_'+d.lower()],axis='columns',inplace=True)

    # clean up contracts. Create BidLvl, BidSuit, Dbl columns.
    contractdf = df['Contract'].str.replace(' ','').str.upper().str.replace('NT','N').str.extract(r'^(?P<BidLvl>\d)(?P<BidSuit>C|D|H|S|N)(?P<Dbl>X*)$')
    # PerformanceWarning: DataFrame is highly fragmented.
    df['BidLvl'] = contractdf['BidLvl']
    # PerformanceWarning: DataFrame is highly fragmented.
    df['BidSuit'] = contractdf['BidSuit']
    # PerformanceWarning: DataFrame is highly fragmented.
    df['Dbl'] = contractdf['Dbl']
    del contractdf
    # There's all sorts of exceptional crap which needs to be done for 'PASS', 'NP', 'BYE', 'AVG', 'AV+', 'AV-', 'AVG+', 'AVG-', 'AVG+/-'. Only 'PASS' is handled, the rest are dropped.
    drop_rows = df['Contract'].ne('PASS')&(df['Score_NS'].eq('PASS')&df['Score_EW'].eq('PASS')&df['BidLvl'].isna()|df['BidSuit'].isna()|df['Dbl'].isna())
    print_to_log(logging.WARNING, 'Invalid contracts: drop_rows:',drop_rows.sum(),df[drop_rows][['Contract','BidLvl','BidSuit','Dbl']])
    df.drop(df[drop_rows].index,inplace=True)
    drop_rows = ~df['Declarer_Direction'].isin(list('NESW')) # keep N,S,E,W. Drop EW, NS, w, ... < 500 cases.
    print_to_log(logging.WARNING, 'Invalid declarers: drop_rows:',drop_rows.sum(),df[drop_rows][['Declarer_Direction']])
    df.drop(df[drop_rows].index,inplace=True)
    df.loc[df['Contract'].ne('PASS'),'Contract'] = df['BidLvl']+df['BidSuit']+df['Dbl']+df['Declarer_Direction']
    df['BidLvl'] = df['BidLvl'].astype('UInt8') # using UInt8 instead of uint8 because of NaNs
    assert (df['Contract'].eq('PASS')|df['BidLvl'].notna()).all()
    assert (df['Contract'].eq('PASS')|df['BidLvl'].between(1,7,inclusive='both')).all()
    assert (df['Contract'].eq('PASS')|df['BidSuit'].notna()).all()
    assert (df['Contract'].eq('PASS')|df['BidSuit'].isin(list('CDHSN'))).all()
    assert (df['Contract'].eq('PASS')|df['Dbl'].notna()).all()
    assert (df['Contract'].eq('PASS')|df['Dbl'].isin(['','X','XX'])).all()

    assert df['Table'].isna().all() or df['Table'].ge(1).all() # some events have NaN table_numbers.
 
    # create more useful Vul column
    # PerformanceWarning: DataFrame is highly fragmented.
    df['iVul'] = df['Board'].map(mlBridgeLib.BoardNumberToVul).astype('uint8') # 0 to 3
    df['Vul'] = df['iVul'].map(lambda x: mlBridgeLib.vul_syms[x]).astype('string') # None, NS, EW, Both
 
    if not pd.api.types.is_numeric_dtype(df['Score_NS']):
        # PerformanceWarning: DataFrame is highly fragmented.
        df['Score_NS'] = df['Score_NS'].astype('string') # make sure all elements are a string
        df.loc[df['Score_NS'].eq('PASS'),'Score_NS'] = '0'
        assert df['Score_NS'].ne('PASS').all()
        drop_rows = ~df['Score_NS'].map(lambda c: c[c[0] == '-':].isnumeric()) | ~df['Score_NS'].map(lambda c: c[c[0] == '-':].isnumeric())
        df.drop(df[drop_rows].index,inplace=True)
        assert df['Score_NS'].isna().sum() == 0
        assert df['Score_NS'].isna().sum() == 0
    df['Score_NS'] = df['Score_NS'].astype('int16')
    df['Score_EW'] = -df['Score_NS']

    # tournaments do not have Tricks or Result columns. Create them.
    # PerformanceWarning: DataFrame is highly fragmented.
    df['scores_l'] = mlBridgeLib.ContractToScores(df) # todo: ValueError: Cannot set a DataFrame with multiple columns to the single column scores_l on
    if 'Result' in df:
        assert df['Result'].notna().all() and df['Result'].notnull().all()
        df['Result'] = df['Result'].map(lambda x: 0 if x in ['=','0',''] else int(x[1:]) if x[0]=='+' else int(x)).astype('int8') # 0 for PASS
    else:
        df['Result'] = df.apply(lambda r: pd.NA if  r['Score_NS'] not in r['scores_l'] else r['scores_l'].index(r['Score_NS'])-(r['BidLvl']+6),axis='columns').astype('Int8') # pd.NA is due to director's adjustment
    if df['Result'].isna().any():
        print_to_log_info('NaN Results:\n',df[df['Result'].isna()][['Board','Contract','BidLvl','BidSuit','Dbl','Declarer_Direction','Score_NS','Score_EW','Result','scores_l']])
    # The following line is on watch. Confirmed that there was an issue with pandas. Effects 'Result' and 'Tricks' and 'BidLvl' columns.
    assert df['Result'].map(lambda x: (x != x) or (x is pd.NA) or -13 <= x <= 13).all() # hmmm, x != x is the only thing which works? Does the new pandas behave as expected? Remove x != x or x is pd.NA?

    if 'Tricks' in df and df['Tricks'].notnull().all(): # tournaments have a Trick column with all None(?).
        assert df['Tricks'].notnull().all()
        df.loc[df['Contract'].eq('PASS'),'Tricks'] = pd.NA
    else:
        df['Tricks'] = df.apply(lambda r: pd.NA if r['Contract'] == 'PASS' else r['BidLvl']+6+r['Result'],axis='columns') # pd.NA is needed for PASS
    if df['Tricks'].isna().any():
        print_to_log_info('NaN Tricks:\n',df[df['Tricks'].isna()][['Board','Contract','BidLvl','BidSuit','Dbl','Declarer_Direction','Score_NS','Score_EW','Tricks','Result','scores_l']])
    df['Tricks'] = df['Tricks'].astype('UInt8')
    # The following line is on watch. Confirmed that there was an issue with pandas. Effects 'Result' and 'Tricks' and 'BidLvl' columns.
    assert df['Tricks'].map(lambda x: (x != x) or (x is pd.NA) or (0 <= x <= 13)).all() # hmmm, x != x is the only thing which works? Does the new pandas behave as expected? Remove x != x or x is pd.NA?

    df['Round'] = df['Round'].fillna(0) # Round is sometimes missing. fill with 0.
    df['tb_count'] = df['tb_count'].fillna(0).astype('uint8') # tb_count is sometimes missing a value. fill with 0.
    df['Table'] = df['Table'].fillna(0).astype('uint8') # todo: Table is often missing a value. fill with 0.

    df.drop(['scores_l'],axis='columns',inplace=True)

    for col in df.columns:
        assert not (col.startswith('ns_') or col.startswith('ew_') or col.startswith('NS_') or col.startswith('EW_')), col

    assert len(df) > 0
    return df.reset_index(drop=True)


# todo: use Augment_Metric_By_Suits or TuplesToSuits?
def Augment_Metric_By_Suits(metrics,metric,dtype='uint8'):
    for d,direction in enumerate(mlBridgeLib.NESW):
        for s,suit in  enumerate(mlBridgeLib.SHDC):
            metrics['_'.join([metric,direction])] = metrics[metric].map(lambda x: x[1][d][0]).astype(dtype)
            metrics['_'.join([metric,direction,suit])] = metrics[metric].map(lambda x: x[1][d][1][s]).astype(dtype)
    for direction in mlBridgeLib.NS_EW:
        metrics['_'.join([metric,direction])] = metrics['_'.join([metric,direction[0]])]+metrics['_'.join([metric,direction[1]])].astype(dtype)
        for s,suit in  enumerate(mlBridgeLib.SHDC):
            metrics['_'.join([metric,direction,suit])] = metrics['_'.join([metric,direction[0],suit])]+metrics['_'.join([metric,direction[1],suit])].astype(dtype)


def TuplesToSuits(df,tuples,column,excludes=[]):
    d = {}
    d['_'.join([column])] = tuples.map(lambda x: x[0])
    for i,direction in enumerate('NESW'):
        d['_'.join([column,direction])] = tuples.map(lambda x: x[1][i][0])
        for j,suit in enumerate('SHDC'):
            d['_'.join([column,direction,suit])] = tuples.map(lambda x: x[1][i][1][j])
    for i,direction in enumerate(['NS','EW']):
        d['_'.join([column,direction])] = tuples.map(lambda x: x[1][i][0]+x[1][i+2][0])
        for j,suit in enumerate('SHDC'):
            d['_'.join([column,direction,suit])] = tuples.map(lambda x: x[1][i][1][j]+x[1][i+2][1][j])
    for k,v in d.items():
        if k not in excludes:
            # PerformanceWarning: DataFrame is highly fragmented.
            df[k] = v
    return d


# Pandas version of mlBridgeLib's Polars version
# Create columns of contract types by partnership by suit by contract. e.g. CT_NS_C_Game
def CategorifyContractTypeByDirection(df):
    contract_types_d = {}
    cols = df.filter(regex=r'CT_(NS|EW)_[CDHSN]').columns
    for c in cols:
        for t in mlBridgeLib.contract_types:
            print_to_log_debug('CT:',c,t,len((t == df[c]).values))
            new_c = c+'_'+t
            contract_types_d[new_c] = (t == df[c]).values
    return contract_types_d


def augment_df(df,sd_cache_d):

    # positions
    df['Pair_Declarer_Direction'] = df['Declarer_Direction'].map(mlBridgeLib.PlayerDirectionToPairDirection)
    df['Opponent_Pair_Direction'] = df['Pair_Declarer_Direction'].map(mlBridgeLib.PairDirectionToOpponentPairDirection)
    df['Direction_OnLead'] = df['Declarer_Direction'].map(mlBridgeLib.NextPosition)
    df['Direction_Dummy'] = df['Direction_OnLead'].map(mlBridgeLib.NextPosition)
    df['Direction_NotOnLead'] = df['Direction_Dummy'].map(mlBridgeLib.NextPosition)
    df['OnLead'] = df.apply(lambda r: r['Player_Number_'+r['Direction_OnLead']], axis='columns') # todo: keep as lower case?
    df['Dummy'] = df.apply(lambda r: r['Player_Number_'+r['Direction_Dummy']], axis='columns') # todo: keep as lower case?
    df['NotOnLead'] = df.apply(lambda r: r['Player_Number_'+r['Direction_NotOnLead']], axis='columns') # todo: keep as lower case?

    # hands
    df['hands'] = df['board_record_string'].map(mlBridgeLib.brs_to_hands)
    assert df['hands'].map(mlBridgeLib.hands_to_brs).eq(df['board_record_string'].str.replace('-','').str.replace('T','10')).all(), df[df['hands'].map(mlBridgeLib.hands_to_brs).ne(df['board_record_string'])][['Board','board_record_string','hands']]
    # ouch. Sometimes acbl hands use '-' in board_record_string, sometimes they don't. Are online hands without '-' and club f-f with '-'? Removing '-' in both so compare works.
    df['PBN'] = df['hands'].map(mlBridgeLib.HandToPBN)
    assert df['PBN'].map(mlBridgeLib.pbn_to_hands).eq(df['hands']).all(), df[df['PBN'].map(mlBridgeLib.pbn_to_hands).ne(df['hands'])]
    brs = df['PBN'].map(mlBridgeLib.pbn_to_brs)
    assert brs.map(mlBridgeLib.brs_to_pbn).eq(df['PBN']).all(), df[brs.map(mlBridgeLib.brs_to_pbn).ne(df['PBN'])]

    # OHE cards
    bin_handsl = mlBridgeLib.HandsLToBin(df['hands'])
    ohe_handsl = mlBridgeLib.BinLToOHE(bin_handsl)
    ohe_hands_df = mlBridgeLib.OHEToCards(df,ohe_handsl)
    df = pd.concat([df,ohe_hands_df],axis='columns',join='inner')

    # hand evaluation metrics
    # todo: use Augment_Metric_By_Suits or TuplesToSuits?
    # 'hands' is ordered CDHS
    hcp = df['hands'].map(mlBridgeLib.HandsToHCP)
    TuplesToSuits(df,hcp,'HCP',['HCP'])
    qt = df['hands'].map(mlBridgeLib.HandsToQT)
    TuplesToSuits(df,qt,'QT',['QT'])
    dp = df['hands'].map(mlBridgeLib.HandsToDistributionPoints)
    TuplesToSuits(df,dp,'DP',['DP'])
    sl = df['hands'].map(mlBridgeLib.HandsToSuitLengths) # sl is needed later by LoTT
    TuplesToSuits(df,sl,'SL',['SL','SL_N','SL_E','SL_S','SL_W','SL_NS','SL_EW'])
    so = mlBridgeLib.CDHS
    for d in mlBridgeLib.NESW:
        # PerformanceWarning: DataFrame is highly fragmented.
        df[f'SL_{d}_{so}'] = df.filter(regex=f'^SL_{d}_[{so}]$').values.tolist() # ordered from clubs to spades [CDHS]
        # PerformanceWarning: DataFrame is highly fragmented.
        df[f'SL_{d}_{so}_J'] = df[f'SL_{d}_{so}'].map(lambda l:'-'.join([str(v) for v in l])).astype('category') # joined CDHS into category
        # PerformanceWarning: DataFrame is highly fragmented.
        df[f'SL_{d}_ML_S'] = df[f'SL_{d}_{so}'].map(lambda l: [v for v,n in sorted([(ll,n) for n,ll in enumerate(l)],key=lambda k:(-k[0],k[1]))]) # ordered most-to-least
        # PerformanceWarning: DataFrame is highly fragmented.
        df[f'SL_{d}_ML_SI'] = df[f'SL_{d}_{so}'].map(lambda l: [n for v,n in sorted([(ll,n) for n,ll in enumerate(l)],key=lambda k:(-k[0],k[1]))]) # ordered most-to-least containing indexes
        # PerformanceWarning: DataFrame is highly fragmented.
        df[f'SL_{d}_ML_SJ'] = df[f'SL_{d}_ML_S'].map(lambda l:'-'.join([str(v) for v in l])).astype('category') # ordered most-to-least and joined into category

    # Create columns containing column names of the NS,EW longest suit.
    sl_cols = [('_'.join(['SL_Max',d]),['_'.join(['SL',d,s]) for s in mlBridgeLib.SHDC]) for d in mlBridgeLib.NS_EW]
    for d in sl_cols:
        # PerformanceWarning: DataFrame is highly fragmented.
        df[d[0]] = df[d[1]].idxmax(axis=1).astype('category') # defaults to object so need string or category

    df = mlBridgeLib.append_double_dummy_results(df)

    # LoTT
    ddmakes = df.apply(lambda r: tuple([tuple([r['_'.join(['DD',d,s])] for s in 'CDHSN']) for d in 'NESW']),axis='columns')
    LoTT_l = [mlBridgeLib.LoTT_SHDC(t,l) for t,l in zip(ddmakes,sl)] # [mlBridgeLib.LoTT_SHDC(ddmakes[i],sl[i]) for i in range(len(df))]
    df['LoTT_Tricks'] = [t for t,l,v in LoTT_l]
    df['LoTT_Suit_Length'] = [l for t,l,v in LoTT_l] # todo: is this correct? use SL_Max_(NS|EW) instead? verify LoTT_Suit_Length against SL_Max_{declarer_pair_direction}.
    df['LoTT_Variance'] = [v for t,l,v in LoTT_l]
    del LoTT_l
    df = df.astype({'LoTT_Tricks':'uint8','LoTT_Suit_Length':'uint8','LoTT_Variance':'int8'})

    # ContractType
    # PerformanceWarning: DataFrame is highly fragmented.
    df['ContractType'] = df.apply(lambda r: 'PASS' if r['Contract'] == 'PASS' else mlBridgeLib.ContractType(r['BidLvl']+6,r['BidSuit']),axis='columns').astype('category')
    # Create column of contract types by partnership by suit. e.g. CT_NS_C.
    contract_types_d = mlBridgeLib.CategorifyContractTypeBySuit(ddmakes)
    contract_types_df = pd.DataFrame(contract_types_d,dtype='category')
    assert len(df) == len(contract_types_df)
    df = pd.concat([df,contract_types_df],axis='columns') # ,join='inner')
    del contract_types_df,contract_types_d
    contract_types_d = CategorifyContractTypeByDirection(df) # using local pandas version instead of mlBridgeLib's Polars version
    contract_types_df = pd.DataFrame(contract_types_d,dtype='category')
    assert len(df) == len(contract_types_df)
    df = pd.concat([df,contract_types_df],axis='columns') # ,join='inner')
    del contract_types_df,contract_types_d

    # create dict of NS matchpoint data.
    matchpoint_ns_d = {} # key is board. values are matchpoint details (score, beats, ties, matchpoints, pct).
    for board,g in df.groupby('Board'):
        board_mps_ns = {}
        for score_ns in g['Score_NS']:
            board_mps_ns = mlBridgeLib.MatchPointScoreUpdate(score_ns,board_mps_ns) # convert to float32 here? It's still a string because it might originally have AVG+ or AVG- etc.
        matchpoint_ns_d[board] = board_mps_ns
    # validate boards are scored correctly
    for board,g in df.groupby('Board'):
        for score_ns,match_points_ns in zip(g['Score_NS'],g['MatchPoints_NS'].astype('float32')):
            if matchpoint_ns_d[board][score_ns][3] != match_points_ns: # match_points_ns is a string because it might originally have AVG+ or AVG- etc.
                print_to_log(logging.WARNING,f'Board {board} score {matchpoint_ns_d[board][score_ns][3]} tuple {matchpoint_ns_d[board][score_ns]} does not match matchpoint score {match_points_ns}') # ok if off by epsilon

    # Vul columns
    df['Vul_NS'] = (df['iVul']&1).astype('bool')
    df['Vul_EW'] = (df['iVul']&2).astype('bool')

    # board result columns
    df['OverTricks'] = df['Result'].gt(0)
    df['JustMade'] = df['Result'].eq(0)
    df['UnderTricks'] = df['Result'].lt(0)

    df[f"Vul_Declarer"] = df.apply(lambda r: r['Vul_'+r['Pair_Declarer_Direction']], axis='columns')
    df['Pct_Declarer'] = df.apply(lambda r: r['Pct_'+r['Pair_Declarer_Direction']], axis='columns')
    df['Pair_Number_Declarer'] = df.apply(lambda r: r['Pair_Number_'+r['Pair_Declarer_Direction']], axis='columns')
    df['Pair_Number_Defender'] = df.apply(lambda r: r['Pair_Number_'+r['Opponent_Pair_Direction']], axis='columns')
    df['Number_Declarer'] = df.apply(lambda r: r['Player_Number_'+r['Declarer_Direction']], axis='columns') # todo: keep as lower case?
    df['Name_Declarer'] = df.apply(lambda r: r['Player_Name_'+r['Declarer_Direction']], axis='columns')
    # todo: drop either Tricks or Tricks_Declarer as they are invariant and duplicates
    df['Tricks_Declarer'] = df['Tricks'] # synonym for Tricks
    df['Score_Declarer'] = df.apply(lambda r: r['Score_'+r['Pair_Declarer_Direction']], axis='columns')
    # recompute Score and compare against actual scores to catch scoring errors such as: Board 1 at https://my.acbl.org/club-results/details/878121
    # just use Score_NS if score is uncomputable probably due to pd.NA from director's adjustment. (r['Result'] is pd.NA) works here. why?
    df['Computed_Score_Declarer'] = df.apply(lambda r: 0 if r['Contract'] == 'PASS' else r['Score_NS'] if r['Result'] is pd.NA else mlBridgeLib.score(r['BidLvl']-1, 'CDHSN'.index(r['BidSuit']), len(r['Dbl']), ('NESW').index(r['Declarer_Direction']), r['Vul_Declarer'], r['Result'],True), axis='columns')
    if (df['Score_Declarer'].ne(df['Computed_Score_Declarer'])|df['Score_NS'].ne(-df['Score_EW'])).any():
        print_to_log(logging.WARNING, 'Invalid Scores:\n',df[df['Score_Declarer'].ne(df['Computed_Score_Declarer'])|df['Score_NS'].ne(-df['Score_EW'])][['Board','Contract','BidLvl','BidSuit','Dbl','Declarer_Direction','Vul_Declarer','Score_Declarer','Computed_Score_Declarer','Score_NS','Score_EW','Result']])
    df['MPs_Declarer'] = df.apply(lambda r: r['MatchPoints_'+r['Pair_Declarer_Direction']], axis='columns')

    df['DDTricks'] = df.apply(lambda r: pd.NA if r['Contract'] == 'PASS' else r['_'.join(['DD',r['Declarer_Direction'],r['BidSuit']])], axis='columns') # invariant
    df['DDTricks_Dummy'] = df.apply(lambda r: pd.NA if r['Contract'] == 'PASS' else r['_'.join(['DD',r['Direction_Dummy'],r['BidSuit']])], axis='columns') # invariant
    # NA for NT. df['DDSLDiff'] = df.apply(lambda r: pd.NA if r['Contract'] == 'PASS' else r['DDTricks']-r['SL_'+r['Pair_Declarer_Direction']+'_'+r['BidSuit']], axis='columns') # pd.NA or zero?
    df['DDScore_NS'] = df.apply(lambda r: 0 if r['Contract'] == 'PASS' else mlBridgeLib.score(r['BidLvl']-1, 'CDHSN'.index(r['BidSuit']), len(r['Dbl']), ('NSEW').index(r['Declarer_Direction']), r['Vul_Declarer'], r['DDTricks']-r['BidLvl']-6), axis='columns')
    df['DDScore_EW'] = -df['DDScore_NS']
    df['DDMPs_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['DDScore_NS'],matchpoint_ns_d[r['Board']])[r['DDScore_NS']][3],axis='columns')
    df['DDMPs_EW'] = df['Board_Top']-df['DDMPs_NS']
    df['DDPct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['DDScore_NS'],matchpoint_ns_d[r['Board']])[r['DDScore_NS']][4],axis='columns')
    df['DDPct_EW'] = 1-df['DDPct_NS']

    # Declarer ParScore columns
    # ACBL online games have no par score data. Must create it.
    if 'par' not in df or df['par'].eq('').all():
        df.rename({'ParScore_EndPlay_NS':'ParScore_NS','ParScore_EndPlay_EW':'ParScore_EW','ParContracts_EndPlay':'ParContracts'},axis='columns',inplace=True)
        #df['ParScore_NS'] = df['ParScore_EndPlay_NS']
        #df['ParScore_EW'] = df['ParScore_EndPlay_EW']
        #df['ParContracts'] = df['ParContracts_EndPlay']
        #df.drop(['ParScore_EndPlay_NS','ParScore_EndPlay_EW','ParContracts_EndPlay'],axis='columns',inplace=True)
    else:
        # parse par column and create ParScore column.
        df['ParScore_NS'] = df['par'].map(lambda x: x.split(' ')[1]).astype('int16')
        df['ParScore_EW'] = -df['ParScore_NS']
        df['ParContracts'] = df['par'].map(lambda x: x.split(' ')[2:]).astype('string')
    if 'par' in df:
        df.drop(['par'],axis='columns',inplace=True)
    df['ParScore_MPs_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['ParScore_NS'],matchpoint_ns_d[r['Board']])[r['ParScore_NS']][3],axis='columns')
    df['ParScore_MPs_EW'] = df['Board_Top']-df['ParScore_MPs_NS']
    df['ParScore_Pct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['ParScore_NS'],matchpoint_ns_d[r['Board']])[r['ParScore_NS']][4],axis='columns')
    df['ParScore_Pct_EW'] = 1-df['ParScore_Pct_NS']
    df["ParScore_Declarer"] = df.apply(lambda r: r['ParScore_'+r['Pair_Declarer_Direction']], axis='columns')
    #df["ParScore_MPs_Declarer"] = df.apply(lambda r: r['ParScore_MPs_'+r['Pair_Declarer_Direction']], axis='columns')
    #df["ParScore_Pct_Declarer"] = df.apply(lambda r: r['ParScore_Pct_'+r['Pair_Declarer_Direction']], axis='columns')
    #df['ParScore_Diff_Declarer'] = df['Score_Declarer']-df['ParScore_Declarer'] # adding convenience column to df. Actual Par Score vs DD Score
    #df['ParScore_MPs_Diff_Declarer'] = df['MPs_Declarer'].astype('float32')-df['ParScore_MPs'] # forcing MPs_Declarer to float32. It is still string because it might originally have AVG+ or AVG- etc.
    #df['ParScore_Pct_Diff_Declarer'] = df['Pct_Declarer']-df['ParScore_Pct_Declarer']
    df['Tricks_DD_Diff_Declarer'] = df['Tricks_Declarer']-df['DDTricks'] # adding convenience column to df. Actual Tricks vs DD Tricks
    #df['Score_DD_Diff_Declarer'] = df['Score_Declarer']-df['DD_Score_Declarer'] # adding convenience column to df. Actual Score vs DD Score

    df['Declarer_Rating'] = df.groupby('Number_Declarer')['Tricks_DD_Diff_Declarer'].transform('mean').astype('float32')
    # todo: resolve naming conflict: Defender_ParScore_GE, Defender_OnLead_Rating, Defender_NotOnLead_Rating vs ParScore_GE_Defender, OnLead_Rating_Defender, NotOnLead_Rating_Defender
    df['Defender_ParScore_GE'] = df['Score_Declarer'].le(df['ParScore_Declarer'])
    df['Defender_OnLead_Rating'] = df.groupby('OnLead')['Defender_ParScore_GE'].transform('mean').astype('float32')
    df['Defender_NotOnLead_Rating'] = df.groupby('NotOnLead')['Defender_ParScore_GE'].transform('mean').astype('float32')

    # masterpoints columns
    for d in mlBridgeLib.NESW:
        df['mp_total_'+d.lower()] = df['mp_total_'+d.lower()].astype('float32')
        df['mp_total_'+d.lower()] = df['mp_total_'+d.lower()].fillna(300) # unknown number of masterpoints. fill with 300.
    df['MP_Sum_NS'] = df['mp_total_n']+df['mp_total_s']
    df['MP_Sum_EW'] = df['mp_total_e']+df['mp_total_w']
    df['MP_Geo_NS'] = df['mp_total_n']*df['mp_total_s']
    df['MP_Geo_EW'] = df['mp_total_e']*df['mp_total_w']

    df, sd_cache_d = Augment_Single_Dummy(df,sd_cache_d,10,matchpoint_ns_d) # {} is no cache

    # todo: check dtypes
    # df = df.astype({'Name_Declarer':'string','Score_Declarer':'int16','ParScore_Declarer':'int16','Pct_Declarer':'float32','DDTricks':'uint8','DD_Score_Declarer':'int16','DD_Pct_Declarer':'float32','Tricks_DD_Diff_Declarer':'int8','Score_DD_Diff_Declarer':'int16','ParScore_DD_Diff_Declarer':'int16','ParScore_Pct_Declarer':'float32','Pair_Declarer':'string','Pair_Defender':'string'})

    # todo: verify every dtype is correct.
    # todo: rename columns when there's a better name
    df.rename({'dealer':'Dealer'},axis='columns',inplace=True)
    assert df['Dealer'].isin(list('NESW')).all()
    df['Dealer'] = df['Dealer'].astype('category') # todo: should this be done earlier?
    assert df['iVul'].isin([0,1,2,3]).all() # 0 to 3
    df['iVul'] = df['iVul'].astype('uint8') # todo: should this be done earlier?
    df['iDate'] = df['Date'].astype('int64')
    return df, sd_cache_d, matchpoint_ns_d


def Augment_Single_Dummy(df,sd_cache_d,produce,matchpoint_ns_d):

    sd_cache_d = mlBridgeLib.append_single_dummy_results(df['PBN'],sd_cache_d,produce)
    df['SDProbs'] = df.apply(lambda r: sd_cache_d[r['PBN']].get(tuple([r['Pair_Declarer_Direction'],r['Declarer_Direction'],r['BidSuit']]),[0]*14),axis='columns') # had to use get(tuple([...]))
    df['SDScores'] = df.apply(Create_SD_Scores,axis='columns')
    df['SDScore_NS'] = df.apply(Create_SD_Score,axis='columns').astype('int16') # Declarer's direction
    df['SDScore_EW'] = -df['SDScore_NS']
    df['SDMPs_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SDScore_NS'],matchpoint_ns_d[r['Board']])[r['SDScore_NS']][3],axis='columns')
    df['SDMPs_EW'] = (df['Board_Top']-df['SDMPs_NS']).astype('float32')
    df['SDPct_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SDScore_NS'],matchpoint_ns_d[r['Board']])[r['SDScore_NS']][4],axis='columns')
    df['SDPct_EW'] = (1-df['SDPct_NS']).astype('float32')
    max_score_contract = df.apply(Create_SD_Score_Max,axis='columns')
    df['SDScore_Max_NS'] = pd.Series([score for score,contract in max_score_contract],dtype='float32')
    df['SDScore_Max_EW'] = pd.Series([-score for score,contract in max_score_contract],dtype='float32')
    df['SDContract_Max'] = pd.Series([contract for score,contract in max_score_contract],dtype='string') # invariant
    del max_score_contract
    df['SDMPs_Max_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SDScore_Max_NS'],matchpoint_ns_d[r['Board']])[r['SDScore_Max_NS']][3],axis='columns')
    df['SDMPs_Max_EW'] = (df['Board_Top']-df['SDMPs_Max_NS']).astype('float32')
    df['SDPct_Max_NS'] = df.apply(lambda r: mlBridgeLib.MatchPointScoreUpdate(r['SDScore_Max_NS'],matchpoint_ns_d[r['Board']])[r['SDScore_Max_NS']][4],axis='columns')
    df['SDPct_Max_EW'] = (1-df['SDPct_Max_NS']).astype('float32')
    df['SDScore_Diff_NS'] = (df['Score_NS']-df['SDScore_NS']).astype('int16')
    df['SDScore_Diff_EW'] = (df['Score_EW']-df['SDScore_EW']).astype('int16')
    df['SDScore_Max_Diff_NS'] = (df['Score_NS']-df['SDScore_Max_NS']).astype('int16')
    df['SDScore_Max_Diff_EW'] = (df['Score_EW']-df['SDScore_Max_EW']).astype('int16')
    df['SDPct_Diff_NS'] = (df['Pct_NS']-df['SDPct_NS']).astype('float32')
    df['SDPct_Diff_EW'] = (df['Pct_EW']-df['SDPct_EW']).astype('float32')
    df['SDPct_Max_Diff_NS'] = (df['Pct_NS']-df['SDPct_Max_NS']).astype('float32')
    df['SDPct_Max_Diff_EW'] = (df['Pct_EW']-df['SDPct_Max_EW']).astype('float32')
    df['SDParScore_Pct_Diff_NS'] = (df['ParScore_Pct_NS']-df['SDPct_Diff_NS']).astype('float32')
    df['SDParScore_Pct_Diff_EW'] = (df['ParScore_Pct_EW']-df['SDPct_Diff_EW']).astype('float32')
    df['SDParScore_Pct_Max_Diff_NS'] = (df['ParScore_Pct_NS']-df['SDPct_Max_Diff_NS']).astype('float32')
    df['SDParScore_Pct_Max_Diff_EW'] = (df['ParScore_Pct_EW']-df['SDPct_Max_Diff_EW']).astype('float32')
    # using same df to avoid the issue with creating new columns. New columns require meta data will need to be changed too.
    sd_df = pd.DataFrame(df['SDProbs'].values.tolist(),columns=[f'SDProbs_Taking_{i}' for i in range(14)])
    for c in sd_df.columns:
        df[c] = sd_df[c].astype('float32')
    return df, sd_cache_d


def Create_SD_Scores(r):
    if r['Contract'] != 'PASS':
        level = r['BidLvl']-1
        suit = r['BidSuit']
        iCDHSN = 'CDHSN'.index(suit)
        nsew = r['Declarer_Direction']
        iNSEW = 'NSEW'.index(nsew)
        vul = r['Vul_Declarer']
        double = len(r['Dbl'])
        scores_l = mlBridgeLib.ScoreDoubledSets(level, iCDHSN, vul, double, iNSEW)
        return scores_l
    else:
        return [0]*14


#def Create_SD_Probs(r):
#    return [r['SDProb_Take_'+str(n)] for n in range(14)] # todo: this was previously computed. can we just use that?


def Create_SD_Score(r):
    probs = r['SDProbs']
    scores_l = r['SDScores']
    ps = sum(prob*score for prob,score in zip(probs,scores_l))
    return ps if r['Declarer_Direction'] in 'NS' else -ps


# Highest expected score, same suit, any level
# Note: score_max may exceed par score when probability of making/setting contract is high.
def Create_SD_Score_Max(r):
    score_max = None
    if r['Contract'] != 'PASS':
        suit = r['BidSuit']
        iCDHSN = 'CDHSN'.index(suit)
        nsew = r['Declarer_Direction']
        iNSEW = 'NSEW'.index(nsew)
        vul = r['Vul_Declarer']
        double = len(r['Dbl'])
        probs = r['SDProbs']
        for level in range(7):
            scores_l = mlBridgeLib.ScoreDoubledSets(level, iCDHSN, vul, double, iNSEW)
            score = sum(prob*score for prob,score in zip(probs,scores_l))
            # todo: do same for redoubled? or is that too rare to matter?
            #scoresx_l = mlBridgeLib.ScoreDoubledSets(level, iCDHSN, vul, 1, iNSEW)
            #scorex = sum(prob*score for prob,score in zip(probs,scoresx_l))
            isdoubled = double
            #if scorex > score:
            #    score = scorex
            #    if isdoubled == 0:
            #        isdoubled = 1
            # must be mindful that NS makes positive scores but EW makes negative scores.
            if nsew in 'NS' and (score_max is None or score > score_max):
                score_max = score
                contract_max = str(level+1)+suit+['','X','XX'][isdoubled]+' '+nsew
            elif nsew in 'EW' and (score_max is None or score < score_max):
                score_max = score
                contract_max = str(level+1)+suit+['','X','XX'][isdoubled]+' '+nsew
    else:
        score_max = 0
        contract_max = 'PASS'
    return (score_max, contract_max)
