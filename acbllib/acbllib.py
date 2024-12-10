# functions which are specific to acbl; downloading acbl webpages, api calls.

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
import re
import traceback
import requests
from bs4 import BeautifulSoup
from io import StringIO
import urllib
from collections import defaultdict
import time
import json
import pathlib
import sys
import sqlalchemy
from sqlalchemy import create_engine, inspect
import sqlalchemy_utils
from sqlalchemy_utils.functions import database_exists, create_database
sys.path.append(str(pathlib.Path.cwd().parent.parent.joinpath('mlBridgeLib'))) # removed .parent
sys.path
import mlBridgeLib


def get_club_results(cns, base_url, acbl_url, acblPath, read_local):
    htmls = {}
    total_clubs = len(cns)
    failed_urls = []
    #headers={"user-agent":None} # Not sure why this has become necessary
    headers={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"}
    for ncn,cn in enumerate(sorted(cns)):
        ncn += 1
        url = base_url+str(cn)+'/'
        file = url.replace(acbl_url,'')+str(cn)+'.html'
        print_to_log_info(f'Processing file ({ncn}/{total_clubs}): {file}')
        path = acblPath.joinpath(file)
        if read_local and path.exists() and path.stat().st_size > 200:
            html = path.read_text(encoding="utf-8")
            print_to_log_info(f'Reading local {file}: len={len(html)}')
        else:
            print_to_log_info(f'Requesting {url}')
            try:
                r = requests.get(url,headers=headers)
            except:
                print_to_log_info(f'Except: status:{r.status_code} {url}')
            else:
                html = r.text
                print_to_log_info(f'Creating {file}: len={len(html)}')
            if r.status_code != 200:
                print_to_log_info(f'Error: status:{r.status_code} {url}')
                time.sleep(60) # obsolete?
                failed_urls.append(url)
                continue
            # pathlib.Path.mkdir(path.parent, parents=True, exist_ok=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            time.sleep(1) # need to self-throttle otherwise acbl returns 403 "forbidden". obsolete?
        htmls[str(cn)] = html
    print_to_log_info(f'Failed Urls: len:{len(failed_urls)} Urls:{failed_urls}')
    print_to_log_info(f"Done: Total clubs processed:{total_clubs}: Total url failures:{len(failed_urls)}")
    return htmls, total_clubs, failed_urls



def extract_club_games(htmls, acbl_url):
    dfs = {}
    ClubInfos = {}
    total_htmls = len(htmls)
    for n,(cn,html) in enumerate(htmls.items()):
        n += 1
        print_to_log_info(f'Processing club ({n}/{total_htmls}) {cn}')
        bs = BeautifulSoup(html, "html.parser") # todo: do this only once.
        html_table = bs.find('table')
        if html_table is None:
            print_to_log_info(f'Invalid club-result for {cn}')
            continue
        # /html/body/div[2]/div/div[2]/div[1]/div[2]
        ClubInfo = bs.find('div', 'col-md-8')
        #print_to_log(ClubInfo)
        ci = {}
        ci['Name'] = ClubInfo.find('h1').contents[0].strip() # get first text and strip
        ci['Location'] = ClubInfo.find('h5').contents[0].strip() # get first text and strip
        if ClubInfo.find('a'):
            ci['WebSite'] = ClubInfo.find('a')['href'] # get href of first a
        ClubInfos[cn] = ci
        print_to_log_info(f'{ci}')
        # assumes first table is our target
        d = pd.read_html(StringIO(str(html_table)))
        assert len(d) == 1
        df = pd.DataFrame(d[0])
        df.insert(0,'Club',cn)
        df.insert(1,'EventID','?')
        hrefs = [acbl_url+link.get('href')[1:] for link in html_table.find_all('a', href=re.compile(r"^/club-results/details/\d*$"))]
        df.drop('Unnamed: 6', axis=1, inplace=True)
        df['ResultID'] = [result.rsplit('/', 1)[-1] for result in hrefs]
        df['ResultUrl'] = hrefs
        dfs[cn] = df
    print_to_log_info(f"Done: Total clubs processed:{len(dfs)}")
    return dfs, ClubInfos    


def extract_club_result_json(dfs, filtered_clubs, starting_nclub, ending_nclub, total_local_files, acblPath,acbl_url, read_local=True):
    total_clubs = len(filtered_clubs)
    failed_urls = []
    total_urls_processed = 0
    total_local_files_read = 0
    #headers={"user-agent":None} # Not sure why this has become necessary. Failed 2021-Sep-02 so using Chrome curl user-agent.
    headers={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"}
    for ndf,(kdf,df) in enumerate(filtered_clubs.items()):
        if ndf < starting_nclub or ndf >= ending_nclub:
            #print_to_log_info(f"Skipping club #{ndf} {kdf}") # obsolete when filtered_clubs works
            continue
        ndf += 1
        except_count = 0
        total_results = len(df['ResultUrl'])
        for cn, (nurl, url) in zip(df['Club'],enumerate(df['ResultUrl'])):
            #nurl += 1
            total_urls_processed += 1
            html_file = url.replace(acbl_url,'').replace('club-results','club-results/'+str(cn))+'.html'
            json_file = html_file.replace('.html','.data.json')
            if nurl % 100 == 0: # commented out because overloaded notebook output causing system instability.
                print_to_log_info(f'Processing club ({ndf}/{total_clubs}): result file ({nurl}/{total_results}): {html_file}')
            #if ndf < 1652:
            #    continue
            html_path = acblPath.joinpath(html_file)
            json_path = acblPath.joinpath(json_file)
            html = None
            data_json = None
            if read_local and json_path.exists():
                #if html_path.exists():
                #    print_to_log(f'Found local html file: {html_file}')
                #else:
                #    print_to_log(f'Missing local html file: {html_file}')
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data_json = json.load(f)
                except:
                    print_to_log_info(f'Exception when reading json file: {json_file}. Deleting html and json files.')
                else:
                    total_local_files_read += 1
                    #print_to_log_info(f'Reading local ({total_local_files_read}/{total_local_files}) file:{json_path}: len:{json_path.stat().st_size}') # commented out because overloaded notebook output causing system instability.
            else:
                print_to_log_info(f'Requesting {url}')
                try:
                    r = requests.get(url,headers=headers)
                except Exception as ex:
                    print_to_log_info(f'Exception: count:{except_count} type:{type(ex).__name__} args:{ex.args}')
                    if except_count > 5:
                        print_to_log_info('Except count exceeded')
                        break # skip url
                    except_count += 1
                    time.sleep(1) # just in case the exception is transient
                    continue # retry url
                except KeyboardInterrupt as e:
                    print_to_log_info(f"Error: {type(e).__name__} while processing file:{url}")
                    print_to_log_info(traceback.format_exc())
                    canceled = True
                    break
                else:
                    except_count = 0            
                html = r.text
                print_to_log_info(f'Creating {html_file}: len={len(html)}')
                # some clubs return 200 (ok) but with instructions to login (len < 200).
                # skip clubs returning errors or tiny files. assumes one failed club result will be true for all club's results.
                if r.status_code != 200 or len(html) < 200:
                    failed_urls.append(url)
                    break
                # pathlib.Path.mkdir(html_path.parent, parents=True, exist_ok=True)
                html_path.parent.mkdir(parents=True, exist_ok=True)
                html_path.write_text(html, encoding="utf-8")
                bs = BeautifulSoup(html, "html.parser")
                scripts = bs.find_all('script')
                #print_to_log(scripts)
                for script in scripts:
                    if script.string: # not defined for all scripts
                        #print_to_log(script.string)
                        vardata = re.search('var data = (.*);\n', script.string)
                        if vardata:
                            data_json = json.loads(vardata.group(1))
                            #print_to_log(json.dumps(data_json, indent=4))
                            print_to_log_info(f"Writing {json_path}")
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(data_json, f, indent=2)
                            bbo_tournament_id = data_json["bbo_tournament_id"]
                            print_to_log_info(f'bbo_tournament_id: {bbo_tournament_id}')
                #time.sleep(1) # obsolete?
            # if no data_json file read, must be an error so delete both html and json files.
            if not data_json:
                html_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)
            #print_to_log(f'Files processed ({total_urls_processed}/{total_local_files_read}/{total_urls_to_process})')
    print_to_log_info(f'Failed Urls: len:{len(failed_urls)} Urls:{failed_urls}')
    print_to_log_info(f"Done: Totals: clubs:{total_clubs} urls:{total_urls_processed} local files read:{total_local_files_read}: failed urls:{len(failed_urls)}")
    return total_urls_processed, total_local_files_read, failed_urls


def club_results_json_to_sql(urls, starting_nfile=0, ending_nfile=0, initially_delete_all_output_files=False, skip_existing_files=True, event_types=[]):
    total_files_written = 0
    if ending_nfile == 0: ending_nfile = len(urls)
    filtered_urls = urls[starting_nfile:ending_nfile]
    total_urls = len(filtered_urls)
    start_time = time.time()

    # delete files first, using filtered list of urls
    if initially_delete_all_output_files:
        for nfile,url in enumerate(filtered_urls):
            sql_file = url.with_suffix('.sql')
            sql_file.unlink(missing_ok=True)

    for nfile,url in enumerate(filtered_urls):
        nfile += 1
        #url = 'https://my.acbl.org/club-results/details/290003' # todo: insert code to extract json from script
        #r = requests.get(url)
        json_file = url
        sql_file = url.with_suffix('.sql')
        print_to_log_info(f"Processing ({nfile}/{total_urls}): file:{json_file.as_posix()}")
        if skip_existing_files:
            if sql_file.exists():
               #print_to_log_info(f"Skipping: File exists:{sql_file.as_posix()}") # removed to avoid too much output
               continue
        try:
            data_json = None
            with open(json_file, 'r', encoding='utf-8') as f:
                data_json = json.load(f)
            #print_to_log(f"Reading {json_file.as_posix()} dict len:{len(data_json)}")
            if len(event_types) > 0 and data_json['type'] not in event_types:
                print_to_log(f"Skipping type:{data_json['type']}: file{json_file.as_posix()}") # removed to avoid too much output
                continue
            tables = defaultdict(lambda :defaultdict(dict))
            primary_keys = ['id']
            mlBridgeLib.json_to_sql_walk(tables,"events","","",data_json,primary_keys) # "events" is the main table.
            with open(sql_file,'w', encoding='utf-8') as f:
                mlBridgeLib.CreateSqlFile(tables,f,primary_keys)
            total_files_written += 1
        except Exception as e:
            print_to_log_info(f"Error: {e}: type:{data_json['type']} file:{url.as_posix()}")
        else:
            print_to_log_info(f"Writing: type:{data_json['type']} file:{sql_file.as_posix()}")

    print_to_log_info(f"All files processed:{total_urls} files written:{total_files_written} total time:{round(time.time()-start_time,2)}")
    return total_urls, total_files_written


# todo: can acblPath be removed?
def club_results_create_sql_db(db_file_connection_string, create_tables_sql_file, db_file_path,  acblPath, db_memory_connection_string='sqlite://', starting_nfile=0, ending_nfile=0, write_direct_to_disk=False, create_tables=True, delete_db=False, perform_integrity_checks=False, create_engine_echo=False):
    if write_direct_to_disk:
        db_connection_string = db_file_connection_string # disk file based db
    else:
        db_connection_string = db_memory_connection_string # memory based db

    if delete_db and sqlalchemy_utils.functions.database_exists(db_file_connection_string):
        print_to_log_info(f"Deleting db:{db_file_connection_string}")
        sqlalchemy_utils.functions.drop_database(db_file_connection_string) # warning: can't delete file if in use by another app (restart kernel).

    if not sqlalchemy_utils.functions.database_exists(db_connection_string):
        print_to_log_info(f"Creating db:{db_connection_string}")
        sqlalchemy_utils.functions.create_database(db_connection_string)
        create_tables = True
        
    engine = sqlalchemy.create_engine(db_connection_string, echo=create_engine_echo)
    raw_connection = engine.raw_connection()

    if create_tables:
        print_to_log_info(f"Creating tables from:{create_tables_sql_file}")
        with open(create_tables_sql_file, 'r', encoding='utf-8') as f:
            create_sql = f.read()
        raw_connection.executescript(create_sql) # create tables

    urls = []
    for path in acblPath.joinpath('club-results').rglob('*.data.sql'): # fyi: PurePathPosix doesn't support glob/rglob
        urls.append(path)

    #urls = [acblPath.joinpath(f) for f in ['club-results/108571/details/280270.data.sql']] # use slashes, not backslashes
    #urls = [acblPath.joinpath(f) for f in ['club-results/275966/details/99197.data.sql']] # use slashes, not backslashes
    #urls = [acblPath.joinpath(f) for f in ['club-results/275966/details/98557.data.sql']] # use slashes, not backslashes
    #urls = [acblPath.joinpath(f) for f in ['club-results/104034/details/100661.data.sql','club-results/104034/details/100663.data.sql']] # use slashes, not backslashes
    #urls = [acblPath.joinpath(f) for f in 100*['club-results/108571/details/191864.data.sql']]

    total_script_execution_time = 0
    total_scripts_executed = 0
    canceled = False
    if ending_nfile == 0: ending_nfile = len(urls)
    filtered_urls = urls[starting_nfile:ending_nfile]
    total_filtered_urls = len(filtered_urls)
    start_time = time.time()
    for nfile,url in enumerate(filtered_urls):
        sql_file = url
        #if (nfile % 1000) == 0:
        #    print_to_log_info(f"Executing SQL script ({nfile}/{total_filtered_urls}): file:{sql_file.as_posix()}")
        
        try:
            sql_script = None
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            start_script_time = time.time()
            raw_connection.executescript(sql_script)
        except Exception as e:
            print_to_log_info(f"Error: {type(e).__name__} while processing file:{url.as_posix()}")
            print_to_log_info(traceback.format_exc())
            print_to_log_info(f"Every json field must be an entry in the schema file. Update schema if needed.")
            print_to_log_info(f"Removing {url.as_posix()}")
            sql_file.unlink(missing_ok=True) # delete any bad files, fix issues, rerun.
            continue # todo: log error.
            #break
        except KeyboardInterrupt as e:
            print_to_log_info(f"Error: {type(e).__name__} while processing file:{url.as_posix()}")
            print_to_log_info(traceback.format_exc())
            canceled = True
            break
        else:
            script_execution_time = time.time()-start_script_time
            if (nfile % 1000) == 0:
                print_to_log_info(f"{nfile}/{total_filtered_urls} SQL script executed: file:{url.as_posix()}: time:{round(script_execution_time,2)}")
            total_script_execution_time += script_execution_time
            total_scripts_executed += 1

    print_to_log_info(f"SQL scripts executed ({total_scripts_executed}/{total_filtered_urls}/{len(urls)}): total changes:{raw_connection.total_changes} total script execution time:{round(time.time()-start_time,2)}: avg script execution time:{round(total_script_execution_time/max(1,total_scripts_executed),2)}")
    # if using memory db, write memory db to disk file.
    if not canceled:
        if perform_integrity_checks:
            # todo: research how to detect and display failures? Which checks are needed?
            print_to_log_info(f"Performing quick_check on file")
            raw_connection.execute("PRAGMA quick_check;") # takes 7m on disk
            print_to_log_info(f"Performing foreign_key_check on file")
            raw_connection.execute("PRAGMA foreign_key_check;") # takes 3m on disk
            print_to_log_info(f"Performing integrity_check on file")
            raw_connection.execute("PRAGMA integrity_check;") # takes 25m on disk
        if not write_direct_to_disk:
            print_to_log_info(f"Writing memory db to file (takes 1+ hours):{db_file_connection_string}")
            engine_file = sqlalchemy.create_engine(db_file_connection_string)
            raw_connection_file = engine_file.raw_connection()
            raw_connection.backup(raw_connection_file.connection) # takes 45m
            raw_connection_file.close()
            engine_file.dispose()
            print_to_log_info(f"Saved {db_file_path}: size:{db_file_path.stat().st_size}")

    raw_connection.close()
    engine.dispose()
    print_to_log_info("Done.")
    return total_scripts_executed # not sure if any return value is needed.


def get_club_results_details_data(url):
    print_to_log_info('details url:',url)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    assert response.status_code == 200, [url, response.status_code]

    soup = BeautifulSoup(response.content, "html.parser")

    if soup.find('result-details-combined-section'):
        data = soup.find('result-details-combined-section')['v-bind:data']
    elif soup.find('result-details'):
        data = soup.find('result-details')['v-bind:data']
    elif soup.find('team-result-details'):
        return None # todo: handle team events
        data = soup.find('team-result-details')['v-bind:data']
    else:
        return None # "Can't find data tag."
    assert data is not None and isinstance(data,str) and len(data), [url, data]

    details_data = json.loads(data) # returns dict from json
    return details_data


def get_club_results_from_acbl_number(acbl_number):
    url = f"https://my.acbl.org/club-results/my-results/{acbl_number}"
    print_to_log_info('my-results url:',url)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    assert response.status_code == 200, [url, response.status_code]

    soup = BeautifulSoup(response.content, "html.parser")

    # Find all anchor tags with href attributes
    anchor_pattern = re.compile(r'/club\-results/details/\d{6}$')
    anchor_tags = soup.find_all('a', href=anchor_pattern)
    anchor_d = {a['href']:a for a in anchor_tags}
    hrefs = sorted(anchor_d.keys(),reverse=True)
    # 847339 2023-08-21, Ft Lauderdale Bridge Club, Mon Aft Stratified Pair, Monday Afternoon, 58.52%
    msgs = [', '.join([anchor_d[href].parent.parent.find_all('td')[i].text.replace('\n','').strip() for i in [0,1,2,3,5]]) for href in hrefs]
    assert len(hrefs) == len(msgs)

    # Print the href attributes
    my_results_details_data = {}
    for href,msg in zip(hrefs,msgs):
        detail_url = 'https://my.acbl.org'+href
        event_id = int(href.split('/')[-1]) # extract event_id from href which is the last part of url
        my_results_details_data[event_id] = (url, detail_url, msg)
    return my_results_details_data


# get a single tournament session result
def get_tournament_session_results(session_id, acbl_api_key):
    headers = {'accept':'application/json', 'Authorization':acbl_api_key[len('Authorization: '):]}
    path = 'https://api.acbl.org/v1/tournament/session'
    query = {'id':session_id,'full_monty':1}
    params = urllib.parse.urlencode(query)
    url = path+'?'+params
    print_to_log_info('tournament session url:',url)
    response = requests.get(url, headers=headers)
    return response


# get a list of tournament session results
def get_tournament_sessions_from_acbl_number(acbl_number, acbl_api_key):
    url, json_responses = download_tournament_player_history(acbl_number, acbl_api_key)
    tournament_sessions_urls = {d['session_id']:(url, f"https://live.acbl.org/event/{d['session_id'].replace('-','/')}/summary", f"{d['date']}, {d['score_tournament_name']}, {d['score_event_name']}, {d['score_session_time_description']}, {d['percentage']}", d) for r in json_responses for d in r['data']} # https://live.acbl.org/event/NABC232/23FP/1/summary
    return tournament_sessions_urls


# get a single player's tournament history
def download_tournament_player_history(player_id, acbl_api_key):
    headers = {'accept':'application/json', 'Authorization':acbl_api_key[len('Authorization: '):]}
    path = 'https://api.acbl.org/v1/tournament/player/history_query'
    query = {'acbl_number':player_id,'page':1,'page_size':200,'start_date':'1900-01-01'}
    params = urllib.parse.urlencode(query)
    url = path+'?'+params
    sessions_count = 0
    except_count = 0
    json_responses = []
    while url:
        try:
            response = requests.get(url, headers=headers)
        except Exception as ex:
            print_to_log_info(f'Exception: count:{except_count} type:{type(ex).__name__} args:{ex.args}')
            if except_count > 5:
                print_to_log_info('Except count exceeded')
                break # skip url
            except_count += 1
            time.sleep(1) # just in case the exception is transient
            continue # retry url
        except KeyboardInterrupt as e:
            print_to_log_info(f"Error: {type(e).__name__} while processing file:{url}")
            print_to_log_info(traceback.format_exc())
            return None
        else:
            except_count = 0
        if response.status_code in [400,500,504]: # 500 is unknown response code. try skipping player
            print_to_log_info(f'Status Code:{response.status_code}: count:{len(json_responses)} skipping') # 4476921 - Thx Merle.
            # next_page_url = None
            # sessions_total = 0
            break
        assert response.status_code == 200, (url, response.status_code) # 401 is authorization error often because Personal Access Token has expired.
        json_response = response.json()
        #json_pretty = json.dumps(json_response, indent=4)
        #print_to_log(json_pretty)
        json_responses.append(json_response)
        url = json_response['next_page_url']
    return path, json_responses


# get a list of player's tournament history
def download_tournament_players_history(player_ids, acbl_api_key, dirPath):
    start_time = time.time()
    get_count = 0 # total number of gets
    #canceled = False
    for n,player_id in enumerate(sorted(player_ids)):
        if player_id.startswith('tmp:') or player_id.startswith('#'): # somehow #* crept into player_id
            print_to_log_info(f'Skipping player_id:{player_id}')
            continue
        else:
            print_to_log_info(f'Processing player_id:{player_id}')
        if dirPath.exists():
            session_file_count = len(list(dirPath.glob('*.session.json')))
            print_to_log_info(f'dir exists: file count:{session_file_count} dir:{dirPath}')
            #if session_file_count == 0: # todo: ignore players who never played a tournament?
            #    print_to_log(f'dir empty -- skipping')
            #    continue
            #if session_file_count > 0: # todo: temp?
            #    print_to_log(f'dir not empty -- skipping')
            #    continue
        else:
            print_to_log_info(f'Creating dir:{dirPath}')
            dirPath.mkdir(parents=True,exist_ok=True)
            session_file_count = 0
        url, json_responses = download_tournament_player_history(player_id, acbl_api_key)
        if json_responses is None: # canceled
            break
        get_count = len(json_responses)
        if get_count == 0: # skip player_id's generating errors. e.g. player_id 5103045, 5103045, 5103053
            continue
        print_to_log_info(f"{n}/{len(player_ids)} gets:{get_count} rate:{round((time.time()-start_time)/get_count,2)} {player_id=}")
        #time.sleep(1) # throttle api calling. Maybe not needed as api is taking longer than 1s.
        sessions_count = 0
        for json_response in json_responses:
            sessions_total = json_response['total'] # is same for every page
            if sessions_total == session_file_count: # sometimes won't agree because identical sessions. revised results?
                print_to_log_info(f'File count correct: {dirPath}: terminating {player_id} early.')
                sessions_count = sessions_total
                break
            for data in json_response['data']:
                sessions_count += 1 # todo: oops, starts first one at 2. need to move
                session_id = data['session_id']
                filePath_sql = dirPath.joinpath(session_id+'.session.sql')
                filePath_json = dirPath.joinpath(session_id+'.session.json')
                if filePath_sql.exists() and filePath_json.exists() and filePath_sql.stat().st_ctime > filePath_json.stat().st_ctime:
                    print_to_log_info(f'{sessions_count}/{sessions_total}: File exists: {filePath_sql}: skipping')
                    #if filePath_json.exists(): # json file is no longer needed?
                    #    print_to_log(f'Deleting JSON file: {filePath_json}')
                    #    filePath_json.unlink(missing_ok=True)
                    break # continue will skip file. break will move on to next player
                if filePath_json.exists():
                    print_to_log_info(f'{sessions_count}/{sessions_total}: File exists: {filePath_json}: skipping')
                    break # continue will skip file. break will move on to next player
                response = get_tournament_session_results(session_id, acbl_api_key)
                assert response.status_code == 200, response.status_code
                session_json = response.json()
                #json_pretty = json.dumps(json_response, indent=4)
                print_to_log_info(f'{sessions_count}/{sessions_total}: Writing:{filePath_json} len:{len(session_json)}')
                with open(filePath_json,'w',encoding='utf-8') as f:
                    f.write(json.dumps(session_json, indent=4))
        if sessions_count != sessions_total:
            print_to_log_info(f'Session count mismatch: {dirPath}: variance:{sessions_count-sessions_total}')
