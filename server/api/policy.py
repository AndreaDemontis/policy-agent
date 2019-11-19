from flask_restplus import Namespace, Resource
from flask import request
import requests, json
# -------------------------------------------------
from server import app
from server.session import Session

# - API Namespace ---------------------------------
ns = Namespace('Policy', description="Policy classificator apis")
# - - - DIALOGFLOW --------------------------------
import dialogflow_v2 as DialogFlow
# -------------------------------------------------

import os
import cPickle
import nltk
import re
import urllib
import requests
from datetime import datetime

from string import lower
from bs4 import *
from server import app
from threading import Thread
from sqlalchemy import exists

db = app.db

DEBUG_MESSAGES = 0
GENERATOR_FOLDER = os.path.join(app.path, 'classifiers/')
CLASSIFIER_FOLDER_LOAD = GENERATOR_FOLDER + 'store/old/' # Folder containing trained classifiers to load
PANDA_TABLE_FOLDER_LOAD = GENERATOR_FOLDER + "panda/old/" # Folder containing the processed dataset to load

THRESHOLD = 0.8 # Probability confidence threshold for positive class

from server.model.paragraph import Paragraph as DB_Paragraph
from server.model.tag_relation import Tag_Relation as DB_Tag_Relation
from server.model.processing_log import Processing_Log as DB_Processing_Log

'''
import scikits.learn
import arff
import html2text
import BeautifulSoup
import requests
import weka
import cPickle
'''

@ns.route('', endpoint="Policy")
class Policy(Resource):

    def post(self):


        data = request.get_json()

        intent = data["queryResult"]["intent"]["displayName"]
        parameters = data["queryResult"]["parameters"]
        context = data["queryResult"]["outputContexts"][-1]["parameters"]

        # - Parameters and context
        service = parameters["service"].capitalize()

        # Database connection, if not found then predict
        '''DATABASE CONNECTION'''

        db_hit = db.session.query(exists().where(DB_Paragraph.service == service)).scalar()
        processing_flag = db.session.query(exists().where(DB_Processing_Log.service == service)).scalar()

        # If service not found
        if not db_hit or processing_flag:
            if not processing_flag:
                log_object = DB_Processing_Log(service=service)
                db.session.add(log_object)
                db.session.commit()
                p = Thread(target=self.pre_elaboration, args=(service,))
                p.start()
            return {
                       "fulfillment_text": "The agent is analyzing the new policy, please wait and ask again in a couple of minutes.",
                       "payload": {}}, 200

        # ------------------------------------------------------------------------------------------
        if intent == "service-request":

            '''QUERY'''
            tag_list = db.session.query(DB_Tag_Relation.tag).join(DB_Paragraph).filter(DB_Paragraph.service == service).distinct(DB_Tag_Relation.tag).all()
            policy_date = db.session.query(DB_Paragraph.policy_date).filter(DB_Paragraph.service == service).first().policy_date.strftime("%Y-%m-%d %H:%M:%S")

            tag_list = map(lambda x: x.tag,tag_list)

            policy_list = []
            for tag in tag_list:
                for substring in ['Contact', 'Location', 'Demographic', 'Identifier']:
                    if(substring not in policy_list and tag.find(substring) == 0):
                        policy_list += [substring]
                if('SSO' not in policy_list and ( tag == 'Facebook_SSO' or tag == 'SSO')):
                    policy_list += ['SSO']

            policy_list = [p.replace('_', ' ') for p in policy_list]

            payload = { "date": policy_date }
            # ------------------------------------------------------------------------------------------
            policy_list = ''.join(e.encode('utf-8') + ', ' for e in policy_list)
            policy_list = policy_list[:-2]
            response = service + " privacy policy has information about the following kind of data: " + policy_list + "."

            return { "fulfillment_text": response, "payload": payload }, 200

        elif intent == "policy-specification":

            # - Parameters and context
            policy = parameters["policy"].capitalize()
            service = context["service"].capitalize()

            # - tell me about contact policy
            # ------------------------------------------------------------------
            tag_list = db.session.query(DB_Tag_Relation.tag).join(DB_Paragraph).filter(DB_Paragraph.service == service).distinct(DB_Tag_Relation.tag).all()

            tag_list = map(lambda x: x.tag, tag_list)

            policy_tag = policy.replace(' ', '_')

            policy_list=[]
            for tag in tag_list:
                if(policy_tag == 'SSO'):
                    if (tag == 'Facebook_SSO' or tag == 'SSO'):
                        policy_list += tag
                elif (tag.find(policy_tag) == 0):
                        policy_list += [tag]

            policy_list_curated = []
            for p in policy_list:
                p = "Other" if p == policy else p
                p = p.replace(policy, '')
                p = p.replace('_', ' ')
                p = p.strip()
                policy_list_curated += [p]

            payload = { }
            # ------------------------------------------------------------------------------------------
            policy_list = ''.join(e.encode('utf-8') + ', ' for e in policy_list_curated)
            policy_list = policy_list[:-2]

            response = service + " privacy policy contains the following '" + policy + "' related tags: " + policy_list

            return { "fulfillment_text": response, "payload": payload }, 200


        elif intent == "policy-specification-text" or intent == "policy-specification-text2":

            # - Parameters and context
            policy = context["policy"].capitalize()
            service = context["service"].capitalize()

            # - give me the full text (about contact policy)
            # ------------------------------------------------------------------
            paragraphs =  db.session.query(DB_Tag_Relation,DB_Paragraph).join(DB_Paragraph).filter(DB_Paragraph.service == service).filter(DB_Tag_Relation.tag == 'Aggregate_' + policy).all()

            paragraphs = map(lambda x: {
                'id': x.Tag_Relation.id,
                'text': x.Paragraph.text,
                'url': x.Paragraph.source_url,
                'positive_feedback': x.Tag_Relation.positive_feedback,
                'negative_feedback': x.Tag_Relation.negative_feedback,
                'confidence': x.Tag_Relation.confidence_level,
            },paragraphs)


            payload = { "policy": paragraphs }
            # ------------------------------------------------------------------------------------------

            response = "This is the list of all paragraphs that contains " + policy + " policy:"

            return { "fulfillment_text": response, "payload": payload }, 200

        elif intent == "policy-specification-subtag":

            # - Parameters and context
            sub = parameters["sub-policy"].capitalize()
            policy = context["policy"].capitalize()
            service = context["service"].capitalize()

            if sub == "Other":
                sub = policy

            # - tell me more about email policy
            # ------------------------------------------------------------------
            paragraphs_count =  db.session.query(DB_Tag_Relation,DB_Paragraph).join(DB_Paragraph).filter(DB_Paragraph.service == service).filter(DB_Tag_Relation.tag == sub).count()

            payload = { }
            # ------------------------------------------------------------------------------------------

            if sub == policy:
                sub = "Other"
            else:
                sub = sub.replace(policy, '')
                sub = sub.replace('_', ' ')
                sub = sub.strip()
                sub = sub.title()

            response = "I found " + str(paragraphs_count) + " paragraphs about " + policy + ": " + sub + " policies."

            return { "fulfillment_text": response, "payload": payload }, 200

        elif intent == "policy-specification-subtag-paragraphs":

            # - Parameters and context
            sub = context["sub-policy"].capitalize()
            policy = context["policy"].capitalize()
            service = context["service"].capitalize()

            if sub == "Other":
                sub = policy

            # - list me all paragraphs
            # ------------------------------------------------------------------
            paragraphs = db.session.query(DB_Tag_Relation,DB_Paragraph).join(DB_Paragraph).filter(DB_Paragraph.service == service).filter(DB_Tag_Relation.tag == sub).all()

            paragraphs = map(lambda x: {
                'id': x.Tag_Relation.id,
                'text': x.Paragraph.text,
                'url': x.Paragraph.source_url,
                'positive_feedback': x.Tag_Relation.positive_feedback,
                'negative_feedback': x.Tag_Relation.negative_feedback,
                'confidence': x.Tag_Relation.confidence_level,
            }, paragraphs)

            payload = { "policy": paragraphs }
            # ------------------------------------------------------------------------------------------

            if sub == policy:
                sub = "Other"
            else:
                sub = sub.replace(policy, '')
                sub = sub.replace('_', ' ')
                sub = sub.strip()
                sub = sub.title()

            response = "This is the list of all paragraphs that contains information on '" + policy + ": " + sub + "' policy:"

            return { "fulfillment_text": response, "payload": payload }, 200

        elif intent == "full-policy":

            # - Parameters and context
            service = context["service"].capitalize()

            # - Give me the full policy (about twitter)
            # ------------------------------------------------------------------
            links = db.session.query(DB_Paragraph.source_url).filter(DB_Paragraph.service == service).distinct()
            links = map(lambda x: x.source_url,links)
            link_list = ''.join(e.encode('utf-8') + u' \n ' for e in links)
            payload = { }
            # ------------------------------------------------------------------------------------------

            response = "You can find the complete policy text following these links: " + link_list

            return { "fulfillment_text": response, "payload": payload }, 200

        return 404

    def pre_elaboration(self,service):

        '''GET POLICY PARAGRAPHS'''

        api_url = 'https://tosdr.org/api/1/service/' + lower(service) + '.json'
        headers = {'Accept-Language': 'en-US,en;q=0.8'}
        api_request = requests.get(api_url,headers=headers)
        policy_date = datetime.now()

        original_paragraphs = []
        if api_request.status_code == 200:
            for policy_link_name in api_request.json()['links']:
                policy_url = api_request.json()['links'][policy_link_name]['url']
                html = urllib.urlopen(policy_url).read()
                soup = BeautifulSoup(html, "html.parser")

                for script in soup(["script", "style"]):
                    script.decompose()

                text = soup.get_text()

                # PROCESS PARAGRAPHS

                text_paragraphs = text.split('\n')

                for current_paragraph in text_paragraphs:
                    if current_paragraph != '' and len(current_paragraph) > 140:
                        current_paragraph = current_paragraph.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(r"\'", "'").replace(r"\"", u"\u0022")
                        current_paragraph = re.sub(r'[^\x00-\x7F]+', '', current_paragraph)
                        original_paragraphs += [{'text': current_paragraph,
                                            'url': policy_url}]


        '''LOAD CLASSIFIERS'''
        classifiers_dict = {}  # Load the classifiers in a dictionary with the format: (Label, Trained Classifier)
        if os.path.isdir(CLASSIFIER_FOLDER_LOAD):  # If the folder containing the classifiers exists
            for filename in os.listdir(CLASSIFIER_FOLDER_LOAD):  # Get each file in the classifier folder
                if filename.endswith(".pkl"):  # Only the pkl files

                    if DEBUG_MESSAGES:
                        print('Loading ' + filename)
                    with open(CLASSIFIER_FOLDER_LOAD + filename, 'rb') as fid:  # Open classifier file
                        classifiers_dict[filename[:-4]] = cPickle.load(fid)  # Load pickle file into the dictionary

        '''STORE PARAGRAPHS '''

        last_paragraph = db.session.query(DB_Paragraph.id).order_by(DB_Paragraph.id.desc()).first()
        if (last_paragraph == None):
            paragraph_id = 1
        else:
            paragraph_id = last_paragraph.id + 1

        for p in original_paragraphs:
            paragraph_object = DB_Paragraph(id=paragraph_id, text=p['text'], service=service, source_url= p['url'],policy_date=policy_date.strftime("%Y-%m-%d %H:%M:%S"))
            db.session.add(paragraph_object)
            db.session.commit()
            paragraph_id = paragraph_id + 1

        '''TEXT PROCESSING'''

        ps = nltk.stem.PorterStemmer()  # Word Stemmer
        nltk.download('stopwords')  # Get stop words
        stop_words = set(nltk.corpus.stopwords.words('english'))  # Set english stop words

        processed_policy = map(lambda x:x['text'],original_paragraphs)

        '''TEXT PROCESSING TASKS'''
        for segment in processed_policy:  # For each paragraph

            processed_text = segment.split()  # Get list of words in paragraph

            '''REMOVE STOP-WORDS'''
            processed_text = filter(lambda x: x not in stop_words,
                                    processed_text)  # Uses previously set english stop words

            '''FILTER DATES, NUMBERS AND URLS'''
            processed_text = filter(
                lambda x: not re.match('^ (?:(?:[0-9]{2}[:\ /, ]){2}[0 - 9]{2, 4} | am | pm)$',
                                       x) and  # https://stackoverflow.com/questions/37473219/how-to-remove-dates-from-a-list-in-python
                          not re.match('/^[\+\-]?\d*\.?\d+(?:[Ee][\+\-]?\d+)?$/',
                                       x) and  # https://stackoverflow.com/questions/9011524/regex-to-check-whether-a-string-contains-only-numbers
                          not re.match('^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$',
                                       x) and  # https://stackoverflow.com/questions/267399/how-do-you-match-only-valid-roman-numerals-with-a-regular-expression
                          not re.match(
                              '[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
                              x)
                # https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
                ,
                processed_text)  # Removes datetimes, numbers, roman numerals and urls in that order

            '''WORD STEMMING'''
            processed_text = map(lambda x: ps.stem(x),
                                 processed_text)  # Uses previously declared porter stemmer ps

            '''REBUILD STRING'''
            segment = ''.join(
                e.encode('utf-8') + ' ' for e in processed_text)  # Paragraph rebuilding using utf-8

        '''TOKENIZING'''
        # Get vectorizer file
        with open(PANDA_TABLE_FOLDER_LOAD + 'vectorizer.pkl', 'rb') as fid:
            vectorizer = cPickle.load(fid)  # Load pickle file
        tokenized_policy = vectorizer.transform(processed_policy)

        '''PREDICT'''
        tag_list = []
        for tag in classifiers_dict.keys():

            classifier = classifiers_dict[tag]

            predictions = classifier.predict(tokenized_policy)

            i = 0
            while i < len(predictions):
                '''''''''''  IMPORTANT AGGREGATE HANDLING'''''''''
                '''''''''''  ID FOR TAG RELATION FOR UPDATE??'''''''''
                if (predictions[i] > THRESHOLD):

                    if (last_paragraph == None):
                        paragraph_id = i + 1
                    else:
                        paragraph_id = last_paragraph.id + i + 1

                    tag_relation_object = DB_Tag_Relation(paragraph=paragraph_id, tag=tag,
                                                          confidence_level=float(predictions[i]))
                    db.session.add(tag_relation_object)
                    db.session.commit()
                i += 1

        log_object =  db.session.query(DB_Processing_Log).filter(DB_Processing_Log.service == service).first()
        db.session.delete(log_object)
        db.session.commit()
