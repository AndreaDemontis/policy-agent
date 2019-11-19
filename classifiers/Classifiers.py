import yaml
import nltk
import re
import pandas
import cPickle
import numpy
import os
import sklearn
import csv

'''
DISCLAIMER

If you use this dataset (APP-350_v1.0) as part of a publication, you must cite the following paper:

MAPS: Scaling Privacy Compliance Analysis to a Million Apps. Sebastian Zimmeck, Peter Story, Daniel Smullen, Abhilasha Ravichander, Ziqi Wang, Joel Reidenberg, N. Cameron Russell, and Norman Sadeh. Privacy Enhancing Technologies Symposium 2019.
'''

'''DIRECTORIES CONFIGURATION'''

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Save folders will be created if they aren't found
DATASET_FOLDER = os.path.join(ROOT_DIR, "APP-350_v1.0/APP-350_v1.0/annotations/") # Folder containing unpacked APP-350_v1.0 dataset's annotations, .yml files
SCORES_SAVE_FOLDER = os.path.join(ROOT_DIR, "scores/") # Save folder for cross-validation scores
CLASSIFIER_FOLDER_LOAD = os.path.join(ROOT_DIR, 'store/old/') # Folder containing trained classifiers to load
CLASSIFIER_FOLDER_SAVE = os.path.join(ROOT_DIR, 'store/new/') # Save location for trained classifiers
PANDA_TABLE_FOLDER_LOAD = os.path.join(ROOT_DIR, "panda/old/") # Folder containing the processed dataset to load
PANDA_TABLE_FOLDER_SAVE = os.path.join(ROOT_DIR, "panda/new/") # Save location for the processed dataset

'''PROGRAM BEHAVIOUR CONFIGURATION'''
DEBUG_MESSAGES = 1 # Print debug messages

PROCESS_DATASET = 1 # Load raw dataset and processes it
PANDA_TABLE_LOAD = 0 # Load pickle file containing the processed dataset's panda table, either this or PROCESS_DATASET have to be set at 1
PANDA_TABLE_SAVE = 1 # Save processed dataset's panda table into a picke file (if PROCESS_DATASET = 1)

USE_CUSTOM_TAGS = 0 # Use custom set of tags, useful for testing or resuming interrupted session

CROSS_VALIDATION = 0 # Perform 10 fold cross-validation
PRINT_SCORES = 1 # Print cross-validation scores
SAVE_SCORES = 1 # Save cross-validation scores in CSV format

TRAIN_CLASSIFIERS = 1 # Train classifiers after cross-validation
LOAD_CLASSIFIERS = 0 # Load trained classifiers from pickle files
SAVE_CLASSIFIERS = 1 # Save trained classifiers into pickle files (if TRAIN_CLASSIFIERS = 1)

'''MODEL CONFIGURATION'''
THRESHOLD = 0.5 # Probability confidence threshold for positive class
from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1) # ATTENTION n_jobs=-1 means ALL processors will be used

# Columns and their order for CSV file containing the cross-validation score
CSV_HEADER = ['tp', 'tn', 'fp', 'fn', 'act_pos', 'tot_pos', 'act_neg', 'tot_neg', 'positive_precision', 'positive_recall', 'positive_f-measure', 'negative_precision', 'negative_recall', 'negative_f-measure', 'precision', 'recall', 'f-measure']

# Custom set of tags if needed for testing (or resuming interrupted sessions)
#TAGS_CUSTOM_SET = ['Identifier_Cookie_or_similar_Tech', 'Identifier_IP_Address', 'Demographic', 'Demographic_Age', 'Contact_E_Mail_Address', 'Contact_Phone_Number', 'Contact_Postal_Address', 'Facebook_SSO', 'SSO', 'Identifier_Device_ID', 'Location', 'Identifier_IMEI', 'Identifier_MAC', 'Identifier_Mobile_Carrier', 'Contact', 'Contact_Address_Book', 'Identifier', 'Identifier_SIM_Serial', 'Location_GPS', 'Location_Bluetooth', 'Location_WiFi', 'Location_Cell_Tower', 'Location_IP_Address', 'Contact_Password', 'Identifier_Ad_ID', 'Identifier_IMSI', 'Contact_City', 'Demographic_Gender', 'Contact_ZIP', 'Identifier_SSID_BSSID', 'Aggregate_Contact', 'Aggregate_Location', 'Aggregate_Demographic', 'Aggregate_Identifier', 'Aggregate_SSO']
TAGS_CUSTOM_SET = ['Identifier_Cookie_or_similar_Tech']

'''PROCESS DATASET'''
if PROCESS_DATASET:
    if DEBUG_MESSAGES:
        print('PROCESS DATASET')

    data_segments = [] # List of segments
    ps = nltk.stem.PorterStemmer() # Word Stemmer
    nltk.download('stopwords') # Get stop words
    stop_words = set(nltk.corpus.stopwords.words('english')) # Set english stop words
    dataset_exist = 0 # Flag to check if the dataset is found

    if os.path.isdir(DATASET_FOLDER): # If the folder containing the dataset exists
        for filename in os.listdir(DATASET_FOLDER): # Get each file in the dataset folder
            if filename.endswith(".yml"): # Only the yml files
                if not dataset_exist: # When the first yml file is found the flag is set for the dataset's existence
                    dataset_exist = 1
                with open(DATASET_FOLDER + filename, 'r') as stream: # Open file stream
                    try:
                        '''LOAD YAML FILE'''
                        data_f = yaml.load(stream, Loader=yaml.CLoader) # Load yml file

                        '''TEXT PROCESSING TASKS'''
                        data_f['segments'] = filter(lambda x: x['annotations'] != [],
                                                  data_f['segments'])  # Remove untagged segments

                        for segment in data_f['segments']: # For each paragraph

                            processed_text = segment['segment_text'].split() # Get list of words in paragraph

                            '''REMOVE STOP-WORDS'''
                            processed_text = filter(lambda x: x not in stop_words,
                                                    processed_text) # Uses previously set english stop words

                            '''FILTER DATES, NUMBERS AND URLS'''
                            processed_text = filter(
                                lambda x: not re.match('^ (?:(?:[0-9]{2}[:\ /, ]){2}[0 - 9]{2, 4} | am | pm)$', x) and # https://stackoverflow.com/questions/37473219/how-to-remove-dates-from-a-list-in-python
                                          not re.match('/^[\+\-]?\d*\.?\d+(?:[Ee][\+\-]?\d+)?$/', x) and  # https://stackoverflow.com/questions/9011524/regex-to-check-whether-a-string-contains-only-numbers
                                          not re.match('^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', x) and # https://stackoverflow.com/questions/267399/how-do-you-match-only-valid-roman-numerals-with-a-regular-expression
                                          not re.match(
                                              '[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
                                              x) # https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
                                ,
                                processed_text) # Removes datetimes, numbers, roman numerals and urls in that order

                            '''WORD STEMMING'''
                            processed_text = map(lambda x: ps.stem(x), processed_text) # Uses previously declared porter stemmer ps

                            '''REBUILD STRING'''
                            segment['segment_text'] = ''.join(e.encode('utf-8') + ' ' for e in processed_text)  # Paragraph rebuilding using utf-8

                        data_segments += data_f['segments'] # Add to segment's list

                    except yaml.YAMLError as exc: # Handles yaml parsing error
                        exit(exc)
    else: # Handles dataset's folder not found case
         exit('Dataset folder not found')

    if not dataset_exist: # Handles case in which no yaml files are found
        exit('No yml file found, dataset not found')

    '''CREATE PANDA TABLE'''
    if DEBUG_MESSAGES:
        print('CREATE PANDA TABLE')
    segment_ready = {'segment_text': map(lambda x: x['segment_text'], data_segments),
                     'annotations': map(lambda x: map(lambda y: y['practice'].replace('_1stParty','').replace('_3rdParty',''), x['annotations']),
                                        data_segments)} # Removed _1stParty and _3rdParty
    df = pandas.DataFrame(segment_ready) # Creates panda table using previously defined structure, additional columns will be added later

    '''TF-IDF VECTOR CREATION'''
    if DEBUG_MESSAGES:
        print('TF-IDF VECTOR CREATION')

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=(1,2)) # Tf-idf Vectorizer with bigrams and unigrams
    tokenized_text = vectorizer.fit_transform(map(lambda x: x['segment_text'], data_segments)) # Applies both fit and transform to segment texts, results in a sparse matrix
    df['tokenized_text'] = map(lambda x: x, tokenized_text) # The resulting sparse matrix is divided into rows

    if PANDA_TABLE_SAVE: # In case we want to save the processed dataset as a panda table in a pickle file
        if not os.path.isdir(PANDA_TABLE_FOLDER_SAVE): # If the folder isn't found it's created
            os.mkdir(PANDA_TABLE_FOLDER_SAVE)
        with open(PANDA_TABLE_FOLDER_SAVE + 'panda.pkl', 'wb') as fid:
            cPickle.dump(df, fid) # The panda table is dumped into the new file
        with open(PANDA_TABLE_FOLDER_SAVE + 'vectorizer.pkl', 'wb') as fid:
            cPickle.dump(vectorizer, fid) # The tf-idf vector is dumped into the new file

'''LOAD PANDA TABLE FROM FILE'''
if PANDA_TABLE_LOAD: # If we want to load the processed dataset from a file
    if DEBUG_MESSAGES:
        print('LOAD PANDA TABLE FROM FILE')
    if not os.path.isdir(PANDA_TABLE_FOLDER_LOAD): # If the folder isn't found it's created
        os.mkdir(PANDA_TABLE_FOLDER_LOAD)
    with open(PANDA_TABLE_FOLDER_LOAD + 'panda.pkl', 'rb') as fid:
        df = cPickle.load(fid) # The panda table is loaded from the file

'''PREPARE ATTRIBUTES FOR CLASSIFIER'''
from scipy.sparse import vstack
X_attributes = vstack(df['tokenized_text']) # The tokenized text is used as the X attribute, the sparse matrix is put back together
# Y_labels is calculated later for each tag

'''GET TAGS'''
tags = []

for sublist in df['annotations']: # Gets each tag from annotations and appends it to the tag's list if it's not already there
    for item in sublist:
        if item not in tags:
            tags.append(item)

'''CREATE COLUMNS FOR EACH TAG'''
for tag in tags:
    df[tag] = df.apply(lambda row: 1 if tag in row.annotations else 0, axis=1) # If the tag is in the rows's annotation list then its value is 1 (0 otherwise)

'''HANDLE AGGREGATE TAGS'''
for tag in ['Contact','Location','Demographic','Identifier']: # Add aggregate columns
    df['Aggregate_' + tag] = df.apply(lambda row: 1 if 0 in map(lambda x: x.find(tag),row.annotations) else 0, axis=1) # If the tag is found as a substring at the start of an annotation in the row's annotation list then the aggregate tag is present (1)
    tags += ['Aggregate_' + tag] # Add to tag list

# Aggregate tag for SSO has a different format
df['Aggregate_SSO'] = df.apply(lambda row: 1 if ('SSO' in row.annotations or 'Facebook_SSO' in row.annotations) else 0, axis=1) # Add aggregate column
tags += ['Aggregate_SSO'] # Add to tag list

if USE_CUSTOM_TAGS:
    tags = TAGS_CUSTOM_SET # Custom set of tag is used if required

'''CROSS_VALIDATION'''
if CROSS_VALIDATION:
    if DEBUG_MESSAGES:
        print('CROSS_VALIDATION')

    # Import scoring support functions
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import  precision_score, recall_score, f1_score
    from scipy import stats

    labels_t = [0, 1] # Negative and positive class

    # Binary thresholding function for probability prediction array
    def threshold_pred(y_pred): return(map(lambda x: 1 if x > THRESHOLD else 0, y_pred))

    # TN, FP, FN, TP scorer function with thresholding of predicted labels
    def tn(y_true, y_pred): return (confusion_matrix(y_true, threshold_pred(y_pred), labels=labels_t)[0, 0])
    def fp(y_true, y_pred): return (confusion_matrix(y_true, threshold_pred(y_pred), labels=labels_t)[0, 1])
    def fn(y_true, y_pred): return (confusion_matrix(y_true, threshold_pred(y_pred), labels=labels_t)[1, 0])
    def tp(y_true, y_pred): return (confusion_matrix(y_true, threshold_pred(y_pred), labels=labels_t)[1, 1])

    # Various metrics defined for testing but not required as they are calculated using TN, FP, FN, TP
    def precision_t(y_true, y_pred): return (precision_score(y_true, threshold_pred(y_pred), average='macro'))
    def recall_t(y_true, y_pred): return (recall_score(y_true, threshold_pred(y_pred), average='macro'))
    def f1_t(y_true, y_pred): return (f1_score(y_true, threshold_pred(y_pred), average='macro'))
    def precision_positive(y_true, y_pred): return (precision_score(y_true, threshold_pred(y_pred), average=None)[1])
    def recall_positive(y_true, y_pred): return (recall_score(y_true, threshold_pred(y_pred), average=None)[1])
    def f1_positive(y_true, y_pred): return (f1_score(y_true, threshold_pred(y_pred), average=None)[1])
    def precision_negative(y_true, y_pred): return (precision_score(y_true, threshold_pred(y_pred), average=None)[0])
    def recall_negative(y_true, y_pred): return (recall_score(y_true, threshold_pred(y_pred), average=None)[0])
    def f1_negative(y_true, y_pred): return (f1_score(y_true, threshold_pred(y_pred), average=None)[0])

    scoring_dict = {
            'tp': make_scorer(tp),
            'tn': make_scorer(tn),
            'fp': make_scorer(fp),
            'fn': make_scorer(fn)
            }

    ''' Scoring dictionary used for testing purposes (The values are grouped, we used them to check if the calculations are correct for groups)
    scoring_dict += {
            'tp': make_scorer(tp),
            'tn': make_scorer(tn),
            'fp': make_scorer(fp),
            'fn': make_scorer(fn),
            'precision' : make_scorer(precision_t),
            'recall' : make_scorer(recall_t),
            'f1_score' : make_scorer(f1_t),
            'precision_positive': make_scorer(precision_positive),
            'recall_positive': make_scorer(recall_positive),
            'f1_positive': make_scorer(f1_positive),
            'precision_negative': make_scorer(precision_negative),
            'recall_negative': make_scorer(recall_negative),
            'f1_negative': make_scorer(f1_negative)
            }
    '''

    if SAVE_SCORES: # If the cross-validation scores are to be saved in a CSV file
        from datetime import datetime
        scoring_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M") # Datetime is taken and used as part of the file name (example: SCORES_2019-10-13_20-56.csv)
        if not os.path.isdir(SCORES_SAVE_FOLDER): # If the scores folder doesn't exist, it's created
            os.mkdir(SCORES_SAVE_FOLDER)
        with open(SCORES_SAVE_FOLDER + 'SCORES_' + scoring_datetime + '.csv', 'w') as scoring_file: # Scoring file is open at the start in order to write the header of the CSV file
            csv_writer = csv.DictWriter(scoring_file, fieldnames=CSV_HEADER, dialect='excel-tab', extrasaction='ignore') # Additional attributes not specified in CSV_HEADER are ignored
            csv_writer.writeheader() # Header is written

    for tag in tags: # Cross-validation is done for each tag
        Y_labels = df[tag].values # Takes the label values for the current tag
        scores = cross_validate(classifier, X_attributes, Y_labels, scoring=scoring_dict, cv=10, return_train_score=True) # 10 fold cross-validation is used

        # We calculated the scores for test, train and even total for testing purposes, only the test values will be printed/saved
        scores_extended = {
            'test': {
                'tp': sum(scores['test_tp']),
                'tn': sum(scores['test_tn']),
                'fp': sum(scores['test_fp']),
                'fn': sum(scores['test_fn'])
            },
            'train':{
                'tp': sum(scores['train_tp']),
                'tn': sum(scores['train_tn']),
                'fp': sum(scores['train_fp']),
                'fn': sum(scores['train_fn'])
            },
            'total':{
                'tp': sum(scores['test_tp'] + scores['train_tp']),
                'tn': sum(scores['test_tn'] + scores['train_tn']),
                'fp': sum(scores['test_fp'] + scores['train_fp']),
                'fn': sum(scores['test_fn'] + scores['train_fn'])
            }
        }

        if PRINT_SCORES: # If the scores have to be printed into the console
            print(tag)
            #print(scores) # Print unprocessed scores

        for type in scores_extended: # For each type the extended scores are calculated (test, train, total)
            scores_extended[type]['tot_neg'] = scores_extended[type]['tn'] + scores_extended[type]['fn'] # Total negative (TN + FN)
            scores_extended[type]['tot_pos'] = scores_extended[type]['tp'] + scores_extended[type]['fp'] # Total positive (TP + FP)
            scores_extended[type]['act_pos'] = scores_extended[type]['tp'] + scores_extended[type]['fn'] # Actual positive (TP + FN)
            scores_extended[type]['act_neg'] = scores_extended[type]['tn'] + scores_extended[type]['fp'] # Actual negative (TN + FP)
            scores_extended[type]['positive_precision'] = float(scores_extended[type]['tp']) / scores_extended[type]['tot_pos'] # Positive class precision ( TP / Total positives)
            scores_extended[type]['positive_recall'] = float(scores_extended[type]['tp']) / scores_extended[type]['act_pos'] # Positive class recall ( TP / Actual positives)
            scores_extended[type]['positive_f-measure'] = 0 if scores_extended[type]['positive_precision'] == 0 or scores_extended[type]['positive_recall'] == 0 else stats.hmean([scores_extended[type]['positive_precision'], scores_extended[type]['positive_recall']]) # Positive class F-Measure, Harmonic mean between precision and recall for positive class
            scores_extended[type]['negative_precision'] = float(scores_extended[type]['tn']) / scores_extended[type]['tot_neg'] # Negative class precision ( TN / Total negatives)
            scores_extended[type]['negative_recall'] = float(scores_extended[type]['tn']) / scores_extended[type]['act_neg'] # Negative class recall ( TN / Actual negatives)
            scores_extended[type]['negative_f-measure'] = 0 if scores_extended[type]['negative_precision'] == 0 or scores_extended[type][
                'negative_recall'] == 0 else stats.hmean([scores_extended[type]['negative_precision'], scores_extended[type]['negative_recall']]) # Negative class F-Measure, Harmonic mean between precision and recall for negative class
            scores_extended[type]['precision'] = numpy.mean([scores_extended[type]['positive_precision'],scores_extended[type]['negative_precision']]) # Total precision, mean between positive precision and negative precision
            scores_extended[type]['recall'] = numpy.mean([scores_extended[type]['positive_recall'],scores_extended[type]['negative_recall']]) # Total recall, mean between positive recall and negative recall
            scores_extended[type]['f-measure'] = numpy.mean([scores_extended[type]['positive_f-measure'],scores_extended[type]['negative_f-measure']]) # Total F-Measure, mean between positive F-Measure and negative F-Measure

            if PRINT_SCORES: # If the scores have to be printed into the console
                #print(type) # If multiple types of score need to be printed
                if type == 'test': # We only print the test scores
                    print(scores_extended[type]) # Print extended scores
        if PRINT_SCORES:
            print('----------------------------') # Console tag delimiter for scores

        if SAVE_SCORES: # If the cross-validation scores are to be saved in a CSV file
            with open(SCORES_SAVE_FOLDER+ 'SCORES_' + scoring_datetime + '.csv', 'a') as scoring_file: # The scores file is opened for each row, so the lengthy process can be interrupted
                csv_writer = csv.DictWriter(scoring_file, fieldnames=CSV_HEADER, dialect='excel-tab',
                                            extrasaction='ignore') # Additional attributes not specified in CSV_HEADER are ignored
                csv_writer.writerow(scores_extended['test']) # Writes the row corresponding to the current classifier's scores

'''TRAIN CLASSIFIERS'''
if TRAIN_CLASSIFIERS:
    if DEBUG_MESSAGES:
        print('TRAIN CLASSIFIERS')

    classifiers_dict = {} # Save the classifiers in a dictionary with the format: (Label, Trained Classifier)

    if SAVE_CLASSIFIERS:
        if not os.path.isdir(CLASSIFIER_FOLDER_SAVE):  # If the folder doesn't exist, it's created
            os.mkdir(CLASSIFIER_FOLDER_SAVE)

    for tag in tags: # A classifier is trained for each tag
        if DEBUG_MESSAGES:
            print('Training ' + tag)
        Y_labels = df[tag].values # Takes the label values for the current tag
        classifiers_dict[tag] = classifier.fit(X_attributes, Y_labels) # Fit classifier

        if SAVE_CLASSIFIERS: # If the classifiers have to be saved into files
            if DEBUG_MESSAGES:
                print('Saving ' + tag)

            with open(CLASSIFIER_FOLDER_SAVE + tag + '.pkl', 'wb') as fid: # Create classifier file
                cPickle.dump(classifier, fid) # Dump trained classifier into a pickle file

'''LOAD CLASSIFIERS'''
if LOAD_CLASSIFIERS:
    if DEBUG_MESSAGES:
        print('LOAD CLASSIFIERS')

    classifiers_dict = {} # Load the classifiers in a dictionary with the format: (Label, Trained Classifier)

    if not os.path.isdir(CLASSIFIER_FOLDER_LOAD): # If the folder doesn't exist, it's created
        os.mkdir(CLASSIFIER_FOLDER_LOAD)
    for tag in tags: # A classifier is loaded for each tag
        if DEBUG_MESSAGES:
            print('Loading ' + tag)
        with open(CLASSIFIER_FOLDER_LOAD + tag + '.pkl', 'rb') as fid: # Open classifier file
            classifiers_dict[tag] = cPickle.load(fid) # Load pickle file into the dictionary
