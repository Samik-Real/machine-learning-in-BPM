import csv
from datetime import datetime
from datetime import timedelta
import pandas as pd
from IPython.display import Image
from keras.initializers import he_normal
from keras.optimizers import Adam

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from tensorflow_core.python.keras.callbacks import EarlyStopping
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
from sklearn.preprocessing import OneHotEncoder
import os
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import keras


import numpy as np
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from keras.layers import Dense, PReLU, BatchNormalization
from keras import Sequential

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import seaborn as sns
import matplotlib.pyplot as plt

from yellowbrick.model_selection import RFECV



class Case:
    def __init__(self, case_ID=None, application_type=None, loan_goal=None, requested_amount=None, activity=None, activity_employee=None, activity_timestamp=None):
        self.activity_array = []
        self.activity_employee_array = []
        self.activity_timestamp_array = []
        self.activity_duration_array = []

        self.total_case_duration = 0

        self.offer_array = []
        self.offer_accepted = False # if any offer was accepted and given
        self.credit_score = float(0)

        self.case_ID = case_ID
        self.application_type = application_type
        self.loan_goal = loan_goal
        self.requested_amount = float(requested_amount)


        self.activity_array.append(activity)
        self.activity_employee_array.append(activity_employee)
        date_time_object = datetime.strptime(activity_timestamp, '%Y-%m-%d %H:%M:%S')
        self.activity_timestamp_array.append(date_time_object)


    def addActivity(self, activity=None, timestamp=None, employee=None, credit_score=None, offer_ID=None, offered_amount=None, offer_monthly_cost=None, offer_first_withdrawal_amount=None, offer_created_timestamp=None, offer_selected=None):
        if activity is not None and timestamp is not None:
            if self.activity_timestamp_array[-1] != timestamp and self.activity_array[-1] != activity: #check if last added activity is the same as activity to add (log file has duplicates)
                self.activity_array.append(activity)
                self.activity_employee_array.append(employee)
                if credit_score is not None and self.credit_score < float(credit_score):
                    self.credit_score = float(credit_score)
                date_time_object = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                self.activity_timestamp_array.append(date_time_object)
                if offer_selected is not None:
                    self.addOffer(offer_ID, offered_amount, offer_monthly_cost, offer_first_withdrawal_amount, offer_selected, offer_created_timestamp)
                    if offer_selected is True:
                        self.offer_accepted = True

    def addOffer(self, offer_ID=None, offered_amount=None, offer_monthly_cost=None, offer_first_withdrawal_amount=None, offer_selected=None, offer_created_timestamp=None):
        self.offer_array.append(Offer(offer_ID, offered_amount, offer_monthly_cost, offer_first_withdrawal_amount, offer_selected, offer_created_timestamp))

    def calculateActivityDurations(self):
        for i in range(len(self.activity_array)):
            if i == 0:
                current_time = datetime.now()
                self.activity_duration_array.append(current_time - current_time)
            elif i > 0:
                self.activity_duration_array.append(self.activity_timestamp_array[i] - self.activity_timestamp_array[i-1])
        self.total_case_duration = self.activity_timestamp_array[-1] - self.activity_timestamp_array[0]

    def getDateTimeActivityTimestamp(self):
        string = ''
        for i in range(len(self.activity_timestamp_array)):
            string = string + ', ' + str(self.activity_timestamp_array[i])
        return string

    def getDateTimeDuration(self):
        string = ''
        for i in range(len(self.activity_timestamp_array)):
            string = str(self.activity_timestamp_array[i]) + ', ' + string
        return string

    def getEndActivity(self):
        return self.activity_array[-1]

    def getNumberOfOffers(self):
        return len(self.offer_array)

    def getNumberOfHandovers(self): #number of times work is passed from one employee to another
        current_employee = ''
        handover_number = 0
        for i in range(len(self.activity_employee_array)):
            if i == 0:
                current_employee = self.activity_employee_array[0]
                continue
            elif self.activity_employee_array[i] != current_employee:
                current_employee = self.activity_employee_array[i]
                handover_number += 1
        normalized_handover = handover_number/len(self.activity_array)
        return handover_number, normalized_handover


    def getSelectedOffer(self):
        string = ''
        for offer in self.offer_array:
            if offer.selected is True:
                string = string + '\n' + '    Offer ID: ' + offer.offer_ID
                string = string + '\n' + '    Offer creation date: ' + str(offer.offer_created_timestamp)
                string = string + '\n' + '    Offer amount: ' + str(offer.offered_amount)
                string = string + '\n' + '    First withdrawal amount: ' + str(offer.first_withdrawal_amount)
                string = string + '\n' + '    Offer cost per month: ' + str(offer.monthly_cost)
                return string, offer.offered_amount, offer.first_withdrawal_amount
        return "No offer was selected", float(0), float(0)

    def toString(self):
        string = "                    *** Case " + str(self.case_ID) + " ***"
        string = string + '\n' + "Case ID: " + str(self.case_ID)
        string = string + '\n' + "Application type: " + str(self.application_type)
        string = string + '\n' + "Number of activities: " + str(len(self.activity_array))
        string = string + '\n' + "Activities in case: " + str(self.activity_array)
        string = string + '\n' + "Employees in activity: " + str(self.activity_employee_array)
        string = string + '\n' + "Number of work handovers: " + str(self.getNumberOfHandovers()[0])
        string = string + '\n' + "Number of work handovers divided by number of activities: " + str(self.getNumberOfHandovers()[1])
        string = string + '\n' + "Timestamps of activities: " + str(self.getDateTimeActivityTimestamp())
        string = string + '\n' + "Duration of activities: " + str(self.activity_duration_array)
        string = string + '\n' + "Total duration of case: " + str(self.total_case_duration)
        string = string + '\n' + "End activity: " + str(self.getEndActivity())
        string = string + '\n' + "Credit Score: " + str(self.credit_score)
        string = string + '\n' + "Loan goal: " + str(self.loan_goal)
        string = string + '\n' + "Number of offers: " + str(self.getNumberOfOffers())
        string = string + '\n' + "Selected offer: " + str(self.getSelectedOffer()[0])
        string = string + '\n' + "                                *** Case End ***"
        return string





class Offer:
    def __init__(self, offer_ID=None, offered_amount=None, monthly_cost=None, first_withdrawal_amount=None, selected=None, offer_created_timestamp=None):
        self.offer_ID = offer_ID
        self.offered_amount = float(offered_amount)
        self.monthly_cost = float(monthly_cost)
        self.first_withdrawal_amount = float(first_withdrawal_amount)
        self.selected = selected
        self.offer_created_timestamp = offer_created_timestamp





class Utility:
    @staticmethod
    def removeCharacters(string):
        string = string.replace('/', '').replace('<', '').replace('>', '').replace('string', '').replace('date', '').replace('"', '').replace('=', '').replace('value', '').replace('key', '').split(" ")
        return string

    @staticmethod
    def removeValues(remove_case_ID=False):
        if remove_case_ID == True:
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None
        elif remove_case_ID == False:
            return None, None, None, None, None, None, None, None, None, None, None, None, None

    @classmethod #Classmethods are static methods that have a self reference (to use other static variables)
    def logReader(cls, file=None):
        if file is None:
            raise Exception('No file name given')
        case_hashTable = {} #hashmap of offer_ID and Case objects
        case_ID, application_type, loan_goal, requested_amount, activity, activity_employee, activity_timestamp, credit_score, offer_ID, offered_amount, offer_monthly_cost, offer_first_withdrawal_amount, offer_selected, offer_created_timestamp = cls.removeValues(True)

        in_trace = False
        in_event = False
        file_reader = open(file)
        for raw_line in tqdm(file_reader):
            line = raw_line.strip()
            if line == "<trace>":
                in_trace = True
            elif line == "</trace>":
                if case_ID is not None:
                    case_hashTable.get(case_ID).calculateActivityDurations()
                case_ID, application_type, loan_goal, requested_amount, activity, activity_employee, activity_timestamp, credit_score, offer_ID, offered_amount, offer_monthly_cost, offer_first_withdrawal_amount, offer_selected, offer_created_timestamp = cls.removeValues(True)
            if in_trace == True:
                if line == "<event>":
                    in_event = True
                elif line == "</event>":
                    if case_ID is not None and case_ID in case_hashTable:
                        case_hashTable.get(case_ID).addActivity(activity, activity_timestamp, activity_employee, credit_score, offer_ID, offered_amount, offer_monthly_cost, offer_first_withdrawal_amount, offer_created_timestamp, offer_selected)
                        application_type, loan_goal, requested_amount, activity, activity_employee, activity_timestamp, credit_score, offer_ID, offered_amount, offer_monthly_cost, offer_first_withdrawal_amount, offer_selected, offer_created_timestamp = cls.removeValues(False)
                    elif case_ID is not None and case_ID not in case_hashTable:
                        case_hashTable.update({case_ID: Case(case_ID, application_type, loan_goal, requested_amount, activity, activity_employee, activity_timestamp)})
                        application_type, loan_goal, requested_amount, activity, activity_employee, activity_timestamp, credit_score, offer_ID, offered_amount, offer_monthly_cost, offer_first_withdrawal_amount, offer_selected, offer_created_timestamp = cls.removeValues(False)
                    in_event = False
                if in_event == True:
                    line = cls.removeCharacters(line)
                    del line[0]
                    if len(line) > 2:
                        line[1:] = [''.join(line[1:])]
                    #print(line)
                    if len(line) > 0:
                        if line[0] == 'EventID':
                            if line[1].startswith('Application_'):
                                case_ID = line[1]
                        if line[0] == '(case)_ApplicationType':
                            application_type = line[1]
                        if line[0] == '(case)_LoanGoal':
                            loan_goal = line[1]
                        if line[0] == '(case)_RequestedAmount':
                            requested_amount = line[1]
                        if line[0] == 'concept:name':
                            activity = line[1]
                        if line[0] == 'org:resource':
                            activity_employee = line[1]
                        if line[0] == 'time:timestamp':
                            aux = line[1].split(".")
                            timestamp = aux[0].replace('T', ' ')
                            activity_timestamp = timestamp
                        if line[0] == 'CreditScore':
                            credit_score = line[1]
                        if line[0] == 'EventID':
                            if line[1].startswith('Offer_'):
                                offer_ID = line[1]
                        if line[0] == 'OfferedAmount':
                            offered_amount = line[1]
                        if line[0] == 'MonthlyCost':
                            offer_monthly_cost = line[1]
                        if line[0] == 'FirstWithdrawalAmount':
                            offer_first_withdrawal_amount = line[1]
                        if line[0] == 'Selected':
                            if line[1] == 'true':
                                offer_selected = True
                            elif line[1] == 'false':
                                offer_selected = False
                            offer_created_timestamp = activity_timestamp
                elif in_event == False:
                    pass
            elif in_trace == False:
                in_event = False
        case_hashTable_statistics = cls.caseStatistics(case_hashTable)
        return case_hashTable, case_hashTable_statistics

    @classmethod
    def caseStatistics(cls, case_hashTable=None):
        string = '                    *** Case Statistics ***'
        string = string + '\n' + "Cases in hashmap: " + str(len(case_hashTable))
        string = string + '\n' + "Total of " + str(len(cls.getDistinctActivities(case_hashTable))) + " different activities: " + str(cls.getDistinctActivities(case_hashTable)) + " in entire hashmap"
        string = string + '\n' + "Total of " + str(len(cls.getEndActivities(case_hashTable)[0])) + " end activities: " + str(cls.getEndActivities(case_hashTable)[0])
        string = string + '\n' + "End activities frequency: " + str(cls.getEndActivities(case_hashTable)[1])
        string = string + '\n' + "Earliest case is: " + str(cls.getEarliestCase(case_hashTable)[0]) + " with case ID: " + str(cls.getEarliestCase(case_hashTable)[1])
        string = string + '\n' + "Oldest case is: " + str(cls.getEarliestCase(case_hashTable)[2]) + " with case ID: " + str(cls.getEarliestCase(case_hashTable)[3])
        string = string + '\n' + "Total of " + str(cls.getWorkload(case_hashTable)[0]) + " employees \n" + "  Workload per employee: \n" + str(cls.getWorkload(case_hashTable)[1])
        string = string + '\n' + "Between 2016-01-01 and 2016-01-02, total of " + str(cls.getWorkload(case_hashTable, datetime(2016, 1, 1), datetime(2016, 1, 2))[0]) + " employees \n" + "  Workload per employee: \n" + str(cls.getWorkload(case_hashTable, datetime(2016, 1, 1), datetime(2016, 1, 2))[1])
        string = string + '\n' + "Cases with A_Cancelled (by applicant), A_Pending (ongoing) or A_Denied (by bank): " + str(cls.getFinishedApplications(case_hashTable)[0])
        string = string + '\n' + "Filtered " + str(cls.getCaseDurationStatistics(case_hashTable, datetime(2016, 1, 1), datetime(2016, 2, 2))[0])
        string = string + '\n' + "All " + str(cls.getCaseDurationStatistics(case_hashTable)[0])
        string = string + '                               *** End ***'
        string = string + '\n' + '\n'
        return string

    @staticmethod
    def getDistinctActivities(case_hashTable):
        activities = []
        for case in case_hashTable:
            for activity in case_hashTable.get(case).activity_array:
                if activity not in activities:
                    activities.append(activity)
        return activities

    @staticmethod
    def getEndActivities(case_hashTable):
        endActivities = []
        end_activities_frequency = []
        for case in case_hashTable:
            if case_hashTable.get(case).activity_array[-1] not in endActivities:
                endActivities.append(case_hashTable.get(case).activity_array[-1])
                end_activities_frequency.append(1)
            elif case_hashTable.get(case).activity_array[-1] in endActivities:
                for i in range(len(endActivities)):
                    if endActivities[i] == case_hashTable.get(case).activity_array[-1]:
                        end_activities_frequency[i] += 1
        return endActivities, end_activities_frequency

    @staticmethod
    def getEarliestCase(case_hashTable):
        earliest_date = datetime(2019, 6, 2, 11, 22, 25)
        earliest_date_ID = None
        oldest_date = datetime(1919, 6, 2, 11, 22, 25)
        oldest_date_ID = None
        for case in case_hashTable:
            if earliest_date > case_hashTable.get(case).activity_timestamp_array[0]:
                earliest_date = case_hashTable.get(case).activity_timestamp_array[0]
                earliest_date_ID = case_hashTable.get(case).case_ID
            if oldest_date < case_hashTable.get(case).activity_timestamp_array[0]:
                oldest_date = case_hashTable.get(case).activity_timestamp_array[0]
                oldest_date_ID = case_hashTable.get(case).case_ID
        return earliest_date, earliest_date_ID, oldest_date, oldest_date_ID



    @staticmethod
    def getWorkload(case_hashTable, earliest_date=None, oldest_date=None):
        class Employee_Workload:
            def __init__(self, activity):
                self.distinct_activity = []
                self.distinct_activity_frequency = []
                self.distinct_activity.append(activity)
                self.distinct_activity_frequency.append(1)

        employee_hashmap = {} #Hashmap of employee_name:Employee_Workload
        string2 = ""
        for case in case_hashTable:
            for activity, employee, timestamp in zip(case_hashTable.get(case).activity_array, case_hashTable.get(case).activity_employee_array, case_hashTable.get(case).activity_timestamp_array):
                if earliest_date is not None and oldest_date is not None and timestamp > oldest_date:
                    continue
                if earliest_date is not None and oldest_date is not None and timestamp < earliest_date:
                    continue
                if employee not in employee_hashmap:
                    employee_hashmap.update({employee: Employee_Workload(activity)})
                elif employee in employee_hashmap:
                    if activity in employee_hashmap.get(employee).distinct_activity:
                        for i in range(len(employee_hashmap.get(employee).distinct_activity)):
                            if activity == employee_hashmap.get(employee).distinct_activity[i]:
                                employee_hashmap.get(employee).distinct_activity_frequency[i] += 1
                    elif activity not in employee_hashmap.get(employee).distinct_activity:
                        employee_hashmap.get(employee).distinct_activity.append(activity)
                        employee_hashmap.get(employee).distinct_activity_frequency.append(1)

        #TODO: Add more queries for employees (poner el tiempo promedio que se demora cada employee en cada task
        # y sacar los employees que mas se demoran en cada task y que menos se demoran)
        for employee in employee_hashmap:
            string2 = string2 + '     Employee ' + str(employee) + ":" + '\n'
            string2 = string2 + '         Total of ' + str(len(employee_hashmap.get(employee).distinct_activity))  + ' distinct activities' + '\n'
            string2 = string2 + '         Distinct activities: ' + str(employee_hashmap.get(employee).distinct_activity) + ":" + '\n'
            string2 = string2 + '         Distinct activities frequency: ' + str(employee_hashmap.get(employee).distinct_activity_frequency) + ":" + '\n'
        string1 = len(employee_hashmap)
        return string1, string2

    @staticmethod
    def getFinishedApplications(case_hashTable):
        completed_cases_hashmap = {}
        A_Cancelled = 0
        A_Pending = 0
        A_Denied = 0
        for case in case_hashTable:
            if 'A_Cancelled' in case_hashTable.get(case).activity_array:
                completed_cases_hashmap.update({case: case_hashTable.get(case)})
                A_Cancelled += 1
            if 'A_Pending' in case_hashTable.get(case).activity_array:
                completed_cases_hashmap.update({case: case_hashTable.get(case)})
                A_Pending += 1
            if 'A_Denied' in case_hashTable.get(case).activity_array:
                completed_cases_hashmap.update({case: case_hashTable.get(case)})
                A_Denied += 1
        string = '\n'
        string = string + "   Total of  " + str(len(completed_cases_hashmap)) + " completed cases" + "\n"
        string = string + "   A_Cancelled: " + str(A_Cancelled) + "\n"
        string = string + "   A_Pending: " + str(A_Pending) + "\n"
        string = string + "   A_Denied: " + str(A_Denied) + "\n"
        return string, completed_cases_hashmap

    @staticmethod
    def getCaseDurationStatistics(case_hashTable, earliest_date=None, oldest_date=None):
        total_duration = timedelta(days=0, seconds=0, microseconds=0)
        counter = 0
        average_duration = 0
        minimum_duration = timedelta(days=500, seconds=0, microseconds=0)
        minimum_duration_caseID = ''
        maximum_duration = timedelta(days=0, seconds=0, microseconds=0)
        maximum_duration_caseID = ''
        duration_array = []
        for case, value in zip(case_hashTable, case_hashTable.values()):
            if earliest_date is not None and oldest_date is not None and case_hashTable.get(case).activity_timestamp_array[0] >= earliest_date and case_hashTable.get(case).activity_timestamp_array[-1] < oldest_date:
                total_duration += value.total_case_duration
                duration_array.append(value.total_case_duration)
                counter += 1
                if value.total_case_duration < minimum_duration:
                    minimum_duration = value.total_case_duration
                    minimum_duration_caseID = value.case_ID
                if value.total_case_duration > maximum_duration:
                    maximum_duration = value.total_case_duration
                    maximum_duration_caseID = value.case_ID
            elif earliest_date is None and oldest_date is None:
                total_duration += value.total_case_duration
                duration_array.append(value.total_case_duration)
                counter += 1
                if value.total_case_duration < minimum_duration:
                    minimum_duration = value.total_case_duration
                    minimum_duration_caseID = value.case_ID
                if value.total_case_duration > maximum_duration:
                    maximum_duration = value.total_case_duration
                    maximum_duration_caseID = value.case_ID
        duration_array.sort()
        median_duration = duration_array[int(len(duration_array)/2)]
        if counter != 0:
            average_duration = total_duration / counter
        else:
            average_duration = 0
        np.empty(len(duration_array))
        for i in range(len(duration_array)):
            duration_array[i] = duration_array[i].total_seconds()
        duration_array = np.array(duration_array)
        standard_deviation = np.array(duration_array) #convertir en date.time de nuevo de to_seconds
        standard_deviation = np.std(standard_deviation)
        std_aux_seconds = datetime(1, 1, 1) + timedelta(seconds=int(standard_deviation))
        standard_deviation = str(std_aux_seconds.day - 1) + ' days, ' + str(std_aux_seconds.hour) + ':' + str(std_aux_seconds.minute) + str(':') + str(std_aux_seconds.second)
        string = 'Case duration statistics: \n'
        if earliest_date is not None and oldest_date is not None:
            string = string + "  Cases between " + str(earliest_date) + " and " + str(oldest_date) + ": \n"
        string = string + "    Total cases: " + str(counter) + "\n"
        string = string + "    Total duration: " + str(total_duration) + "\n"
        string = string + "    Average case duration: " + str(average_duration) + "\n"
        string = string + "    Standard deviation for case duration: " + str(standard_deviation) + " or " + str(std_aux_seconds) + " seconds" + "\n"
        string = string + "    Median case duration: " + str(median_duration) + "\n"
        string = string + "    Minimum case duration: " + str(minimum_duration) + " with case ID: " + str(minimum_duration_caseID) + "\n"
        string = string + "    Maximum case duration: " + str(maximum_duration) + " with case ID: " + str(maximum_duration_caseID) + "\n"
        return string, counter, total_duration, average_duration, minimum_duration, maximum_duration, median_duration

    @staticmethod
    def logPrinter(file=None):
        if file is None:
            raise Exception('No file name given')
        file_reader = open(file)
        for raw_line in tqdm(file_reader):
            line = raw_line.strip()
            print(line)



class CSV:
    @staticmethod
    def writeCSV(hashmap=None, fileName='CSV.csv', earliestTimeStamp=datetime(2016,1,1,10,51,15)):
        header = ['Case ID', 'Application Type', 'Total Case Duration', 'Loan Goal', 'Requested Amount', 'Offered Amount', 'First Withdrawal Amount', 'Credit Score', 'Number of Offers', 'Number of Handovers', 'Normalized Handovers', 'Number of activities', 'Offer Selected', 'On-time', 'Application Status']
        # ['Offer Selected', 'On-time', 'Application Status'] these are possible 'Y' variables
        hashmap = Utility.getFinishedApplications(hashmap)[1]
        average_duration = Utility.getCaseDurationStatistics(hashmap)[3]
        median_duration = Utility.getCaseDurationStatistics(hashmap)[6]
        with open(fileName, 'wt') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(header)
            row = []
            for case in tqdm(hashmap):
                row.insert(0, hashmap.get(case).case_ID)
                row.insert(1, hashmap.get(case).application_type)
                row.insert(2, hashmap.get(case).total_case_duration.total_seconds())
                row.insert(3, hashmap.get(case).loan_goal)
                row.insert(4, hashmap.get(case).requested_amount)
                row.insert(5, hashmap.get(case).getSelectedOffer()[1])
                row.insert(6, hashmap.get(case).getSelectedOffer()[2])
                row.insert(7, hashmap.get(case).credit_score)
                row.insert(8, len(hashmap.get(case).offer_array))
                row.insert(9, hashmap.get(case).getNumberOfHandovers()[0])
                row.insert(10, hashmap.get(case).getNumberOfHandovers()[1])
                row.insert(11, len(hashmap.get(case).activity_array))
                if hashmap.get(case).getSelectedOffer()[1] > 0:
                    row.insert(12, 1)
                else:
                    row.insert(12, 0)
                date_time_object1 = datetime.strptime('2016-01-01 10:51:15', '%Y-%m-%d %H:%M:%S')
                date_time_object2 = datetime.strptime('2016-01-11 10:51:15', '%Y-%m-%d %H:%M:%S')
                if hashmap.get(case).total_case_duration <= average_duration:    #for average duraton as On-time attribute
                #if hashmap.get(case).total_case_duration <= median_duration:       #for median duraton as On-time attribute
                #if hashmap.get(case).total_case_duration <= (date_time_object2-date_time_object1):       # for custom duraton as On-time attribute
                    row.insert(13, 1)
                else:
                    row.insert(13, 0)
                if 'A_Cancelled' in hashmap.get(case).activity_array:
                    row.insert(14, 0)
                elif 'A_Pending' in hashmap.get(case).activity_array:
                    row.insert(14, 1)
                elif 'A_Denied' in hashmap.get(case).activity_array:
                    row.insert(14, 2)
                else:
                    row.insert(14, -1)
                csv_writer.writerow(row)
                row.clear()

#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
    class One_Hot_Encoder:
        def __init__(self, file='CSV.csv', attribute_names_to_encode=None):
            if attribute_names_to_encode is None:
                self.attribute_names_to_encode = ['Application Type', 'Loan Goal']
            else:
                self.attribute_names_to_encode = attribute_names_to_encode
            self.encoder = OneHotEncoder()
            self.dataset = pd.read_csv(file)
            self.file_name = file
        def fit_data_to_oneHot_and_save_CSV(self):
            x_toEncode = self.dataset[self.attribute_names_to_encode]
            self.encoder.fit(x_toEncode)
            print('Encoded categories: ' + str(self.encoder.categories_))
            x_toEncode = self.encoder.transform(x_toEncode).toarray() #Convert to oneHot vector
            print('Encoded feature names: ' + str(self.encoder.get_feature_names(self.attribute_names_to_encode))) #get the encoded feature names

            with open(self.file_name + '_temporary.csv', 'wt') as open_file:
                csv_writer = csv.writer(open_file)
                csv_writer.writerow(self.encoder.get_feature_names(self.attribute_names_to_encode))
                for i in range(len(x_toEncode)):
                    csv_writer.writerow(x_toEncode[i])

            aux_csv = pd.read_csv(self.file_name + '_temporary.csv')
            os.remove(self.file_name + '_temporary.csv')
            final_X = self.dataset.drop(columns=self.attribute_names_to_encode)
            for i in range(len(self.encoder.get_feature_names(self.attribute_names_to_encode))):
                header = self.encoder.get_feature_names(self.attribute_names_to_encode)[i]
                final_X.insert(i+1, header, aux_csv[[header]], True)
            final_file_name = self.file_name.replace('.csv', '_modified.csv')
            final_X.to_csv(final_file_name, index=False)
            return final_file_name


        def inverse_oneHot_to_data_and_save_CSV(self):
            #TODO: Create inverse function
            print('Inverse function coming up')









#https://www.datacamp.com/community/tutorials/feature-selection-python
#https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
#https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
class FeatureSelection:

    class Filter:
        def __init__(self, file='CSV.csv'):
            self.dataset = pd.read_csv(file)

        def correlationMatrix(self):
            correlation_matrix = self.dataset.corr(method='spearman')
            plt.figure(figsize=(17, 16))
            sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds, linewidths=0.5, fmt='.2f')
            plt.show()
            return correlation_matrix


        # Spearmans Correlation for identifying nonlinear relationships
        def spearmansCorrelation(self, x_columns=None, y_column=None, remove_x_columns=False, num_of_top_values=5):  # no linearity relationship or asumed distribution!
            x = ''
            y = ''
            if remove_x_columns == True:
                columns_to_remove = x_columns
                x = self.dataset.drop(columns=columns_to_remove)
                y = self.dataset[y_column]
            elif remove_x_columns == False:
                x = self.dataset[x_columns]
                y = self.dataset[y_column]

            if num_of_top_values == 'all':
                num_of_top_values = len(x.columns)

            x_column_names = x.columns.values
            y_column_name = y.columns.values
            top_correlations = [0] * num_of_top_values
            top_correlation_attribute_names = ['NA'] * num_of_top_values
            top_p_values = [0] * num_of_top_values
            print("Spearman's Correlation: ")
            for feature_name in x_column_names:
                correlation, p_value = spearmanr(x[feature_name], y)
                print('     correlation for "' + str(y_column_name[0]) + '" and "' + str(feature_name) + '": ' + str(
                    correlation))
                print('     p-value: for "' + str(y_column_name[0]) + '" and "' + str(feature_name) + '": ' + str(
                    p_value))
                print()
                top_correlations, top_correlation_attribute_names, top_p_values = FeatureSelection.Filter._bubbleSortHighestValueFirst( top_correlations, top_correlation_attribute_names, top_p_values, correlation, feature_name, p_value, num_of_top_values)
            print('Top ' + str(num_of_top_values) + ' highest correlation values with class variable "' + str(y_column_name[0]) + '": ')
            print("     attribute names: " + str(top_correlation_attribute_names))
            print("     correlation: " + str(top_correlations))
            print("     p_value: " + str(top_p_values))
            return top_correlation_attribute_names

        @staticmethod
        def _bubbleSortHighestValueFirst(correlation, attribute_name, p_value, add_corr, add_name, add_p_value,
                                         num_of_values):
            correlation.append(add_corr)
            attribute_name.append(add_name)
            p_value.append(add_p_value)
            length = len(correlation)
            for passnumber in range(0, length):
                for i in range((length - 1), 0, -1):
                    if abs(correlation[i]) > abs(correlation[i - 1]):
                        temp = correlation[i]
                        temp2 = attribute_name[i]
                        temp3 = p_value[i]
                        correlation[i] = correlation[i - 1]
                        attribute_name[i] = attribute_name[i - 1]
                        p_value[i] = p_value[i - 1]
                        correlation[i - 1] = temp
                        attribute_name[i - 1] = temp2
                        p_value[i - 1] = temp3
            if len(correlation) > num_of_values:
                del correlation[-1]
                del attribute_name[-1]
                del p_value[-1]
            return correlation, attribute_name, p_value

    class Wrapper:

        def __init__(self, file='CSV.csv'):
            self.dataset = pd.read_csv(file)

        def recursiveFeatureElimination(self, x_columns=None, y_column=None, number_of_features=5, features_removed_per_step=1):
            x = self.dataset[x_columns]
            x = self.dataset[['Normalized Handovers', 'Credit Score']]
            y = self.dataset[y_column]
            classifier = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=2, random_state=11)
            selector = RFE(classifier, n_features_to_select=number_of_features, step=features_removed_per_step)
            y = y.to_numpy()
            selector = selector.fit(x, y.ravel())
            print("Selected Features: " + str(selector.support_))
            print("Feature Ranking: " + str(selector.ranking_))

        #https://www.scikit-yb.org/en/latest/api/model_selection/rfecv.html
        #Go from 1 feature to all features
        def recursiveFeatureEliminationWithCrossValidation(self, x_columns=None, y_column=None, cross_validation=10,features_removed_per_step=1):
            x = self.dataset[x_columns]
            y = self.dataset[y_column]
            classifier = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, random_state=11)
            rfecv = RFECV(classifier, cv=StratifiedKFold(cross_validation), scoring='accuracy', step=features_removed_per_step)
            y = y.to_numpy()
            rfecv = rfecv.fit(x, y.ravel())
            rfecv.show()
            print("Feature names: " + str(x_columns))
            print("Selected Features: " + str(rfecv.support_))
            print("Feature Ranking: " + str(rfecv.ranking_))
            print("Optimal number of features : %d" % rfecv.n_features_)
            #print("Coef importance of selected features: " + str(np.absolute(rfecv.estimator_.coef_)))
            print("Feature importance of selected features: " + str(rfecv.estimator_.feature_importances_))
            featureSelected = rfecv.support_
            selected_features = []
            for i in range(len(featureSelected)):
                if featureSelected[i] == True:
                    selected_features.append(x_columns[i])
            return selected_features


        #TODO: THIS is not working
        # http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
        def sequentialFeatureSelector(self, x_columns=None, y_column=None):
            x = self.dataset[x_columns]
            x = self.dataset[['Normalized Handovers', 'Credit Score']]
            y = self.dataset[y_column]
            classifier = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=2, random_state=11)
            sequential_feature_selector = SFS(classifier, k_features=1, forward=False, floating=False, verbose=2, scoring='accuracy', cv=0, n_jobs=-1)
            #sequential_feature_selector = sequential_feature_selector.fit(x, y, custom_feature_names=x_columns)
            sequential_feature_selector = sequential_feature_selector.fit(x, y, custom_feature_names=['Normalized Handovers', 'Credit Score'])
            print("Feature subsets selected at each step: " + str(sequential_feature_selector.subsets_))
            print("Indices of best features: " + str(sequential_feature_selector.k_feature_idx_))
            print("Accuracy with best features: " + str(sequential_feature_selector.k_score_))
            print("All metrics: " + str(pd.DataFrame.from_dict(sequential_feature_selector.get_metric_dict()).T))





#https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512
class Classification:
    def __init__(self, file='CSV.csv', class_type='Application Status'):
        self.dataset = pd.read_csv(file)
        self.feature_names = ['']
        self.num_classes = 0
        if class_type == 'On-time':
            self.num_classes = 2  # ['0', '1']
        elif class_type == 'Application Status':
            self.num_classes = 3  # classes ['0', '1', '2']
        elif class_type == 'Offer Selected':
            self.num_classes = 2  # classes ['0', '1']
        else:
            raise Exception('"{}" in Classification constructor is not a recognized class type'.format(class_type))

    def classify(self, x_columns=None, y_column=None, remove_x_columns=False, classification_type='decision_tree'):
         x = ''
         y = ''
         if remove_x_columns == True:
             columns_to_remove = x_columns
             x = self.dataset.drop(columns=columns_to_remove)
             y = self.dataset[y_column]
         elif remove_x_columns == False:
             x = self.dataset[x_columns]
             y = self.dataset[y_column]

         x_column_names = x.columns.values
         self.feature_names = x_column_names
         y_class_name = y.columns.values
         scores = []
         if classification_type == 'decision_tree':
             scores = self.decisionTree(x, y, feature_names=x_column_names, class_type=y_class_name)
             print("Decision Tree")
             print("Accuracy: %0.2f (+/- %0.2f)" % (scores[0].mean(), scores[0].std() * 2))
             print("Total scores: " + str(scores[0]))
             if len(scores[1]) > 0:
                 print("AUC: %0.2f (+/- %0.2f)" % (scores[1].mean(), scores[1].std() * 2))
                 print("Total AUC scores: " + str(scores[1]))
         elif classification_type == 'multilayer_perceptron':
             scores = self.multiLayerPerceptron(x, y)
             print("Multilayer Perceptron")
             print("Accuracy: %0.2f (+/- %0.2f)" % (scores[0].mean(), scores[0].std() * 2))
             print("Total scores: " + str(scores[0]))
             if len(scores[1]) > 0:
                 print("AUC: %0.2f (+/- %0.2f)" % (scores[1].mean(), scores[1].std() * 2))
                 print("Total AUC scores: " + str(scores[1]))
         elif classification_type == 'random_forrest':
             scores = self.randomForrest(x, y, feature_names=x_column_names, class_type=y_class_name)
             print("Random Forrest")
             print("Accuracy: %0.2f (+/- %0.2f)" % (scores[0].mean(), scores[0].std() * 2))
             print("Total scores: " + str(scores[0]))
             if len(scores[1]) > 0:
                 print("AUC: %0.2f (+/- %0.2f)" % (scores[1].mean(), scores[1].std() * 2))
                 print("Total AUC scores: " + str(scores[1]))
         else:
             raise Exception('"{}" is not a recognized classification algorithm'.format(classification_type))




    #https://scikit-learn.org/0.15/auto_examples/plot_roc_crossval.html

    #https://www.datacamp.com/community/tutorials/decision-tree-classification-python
    #https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
    #https://scikit-learn.org/stable/modules/cross_validation.html
    def decisionTree(self, x, y, feature_names=None, class_type=None):
        classifier = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, random_state=11)
        classifier = classifier.fit(x, y)
        scores = cross_val_score(classifier, x, y, cv=10, scoring='accuracy') #10 fold cross validation
        auc = []
        if self.num_classes == 2:
            y = y.to_numpy()
            auc = cross_val_score(classifier, x, y.ravel(), cv=10, scoring='roc_auc')  # 10 fold cross validation

        if class_type == 'On-time':
            class_names = ['Overtime', 'On-Time'] # converted from classes ['0', '1']
        elif class_type == 'Application Status':
            class_names = ['App_Cancelled', 'App_Pending', 'App_Denied']  # converted from classes ['0', '1', '2']
        elif class_type == 'Offer Selected':
            class_names = ['Offer_not_selected', 'Offer_selected']  # converted from classes ['0', '1']
        else:
            raise Exception('"{}" is not a recognized class type'.format(class_type))
        dot_data = StringIO()
        export_graphviz(classifier, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_names, class_names=class_names) #class_names enumerates classes starting from class 0 and incrementing class number (no idea how it works with classes that are strings)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('loan_application_decision_tree.png')
        Image(graph.create_png())
        return scores, auc



    #https://chrisalbon.com/deep_learning/keras/k-fold_cross-validating_neural_networks/
    #https://stackoverflow.com/questions/54478702/typeerror-call-missing-1-required-positional-argument-inputs/54479335
    def multiLayerPerceptron(self, x, y):
        neural_network = KerasClassifier(build_fn=self.create_network, epochs=10, batch_size=100,  verbose=1) #create_network needs to be passed without parenthesis (KerasClassifier will run it), which is a callable function   #https://treyhunner.com/2019/04/is-it-a-class-or-a-function-its-a-callable/
        scores = cross_val_score(neural_network, x, y, cv=10, scoring='accuracy') #10 fold cross validation
        auc = []
        if self.num_classes == 2:
            y = y.to_numpy()
            auc = cross_val_score(neural_network, x, y.ravel(), cv=10, scoring='roc_auc')  # 10 fold cross validation
        return scores, auc
    #https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    def create_network(self): #This is using keras to access tensorflow in the background
        model = Sequential()
        model.add(Dense(12, input_dim=len(self.feature_names), activation='linear', kernel_initializer=he_normal(seed=11)))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dense(8, activation='linear', kernel_initializer=he_normal(seed=11)))
        model.add(BatchNormalization())
        model.add(PReLU())
        #model.add(Dense(5, activation='linear', kernel_initializer=he_normal(seed=11)))
        #model.add(BatchNormalization())
        #model.add(PReLU())
        model.add(Dense(self.num_classes, activation='softmax', kernel_initializer=he_normal(seed=11)))
        adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy']) #sparse_categorical_crossentropy does not need labels to be one-hot-encoded but needs last layer to have neuron number equal to number of classes
        return model



    #https://www.datacamp.com/community/tutorials/random-forests-classifier-python
    #https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
    #https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
    #numpy.ravel() is to convert a 2 d array to a 1 d array
    #https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    def randomForrest(self, x, y, feature_names=None, class_type=None):
        classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5, bootstrap=True, max_features='sqrt', random_state=11) #  max_features='sqrt' uses sqrt(n_features) number of features in each forrest
        y = y.to_numpy()
        classifier = classifier.fit(x, y.ravel())
        scores = cross_val_score(classifier, x, y.ravel(), cv=10, scoring='accuracy')  # 10 fold cross validation
        auc = []
        if self.num_classes == 2:
            auc = cross_val_score(classifier, x, y.ravel(), cv=10, scoring='roc_auc')  # 10 fold cross validation

        if class_type == 'On-time':
            class_names = ['Overtime', 'On-Time'] # converted from classes ['0', '1']
        elif class_type == 'Application Status':
            class_names = ['App_Cancelled', 'App_Pending', 'App_Denied']  # converted from classes ['0', '1', '2']
        elif class_type == 'Offer Selected':
            class_names = ['Offer_not_selected', 'Offer_selected']  # converted from classes ['0', '1']
        else:
            raise Exception('"{}" is not a recognized class type'.format(class_type))

        estimator = classifier.estimators_[5]
        dot_data = StringIO()
        export_graphviz(estimator, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_names, class_names=class_names)  # class_names enumerates classes starting from class 0 and incrementing class number (no idea how it works with classes that are strings)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('loan_application_random_forrest_tree_5.png')
        Image(graph.create_png())

        return scores, auc




