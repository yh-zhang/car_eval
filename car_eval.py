# Predict car condition with car features using SVM
# Data source: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

from collections import Counter
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def main():
    trainingData, trainingLabel = processData('./car.data')
    nfoldCrossValidation(trainingData, trainingLabel, './output.txt', samplingMethod=1)

def processData(inputAdr, **kwargs):
    '''
    Parse the raw data into features and labels
    Input:
    inputAdr (str): address of input data file
    return:
    car_feature [[int]]: convert categorical feature into one hot key vector
    car_label [int]: label of car condtions
    '''
    delimiter = ',' if 'delimiter' not in kwargs else kwargs['delimiter']
    label_index = 6 if 'label_index' not in kwargs else kwargs['label_index']
    label_dict ={'unacc':0, 'acc':1,'good':2, 'vgood':3}

    feature_dict = ['0vhigh','0high', '0med', '0low', '1vhigh','1high', '1med', '1low', '22', '23', '24', \
                    '25more', '32', '34', '3more','4small','4med', '4big', '5low', '5med','5high']

    feature_dict = {val:ind for ind, val in enumerate(feature_dict)}

    # Parse the categorical feature with one hot key
    car_feature = []
    car_label = []
    with open(inputAdr) as f:
        for row in f:
            row_content = row.strip().split(delimiter)
            car_label.append(label_dict[row_content[label_index]])
            featureSingleCar = [0] * len(feature_dict)
            for i in range(len(row_content)-1):
                feature_index = str(i)+row_content[i]
                featureSingleCar[feature_dict[feature_index]] = 1
            car_feature.append(featureSingleCar)
    return car_feature, car_label

def nfoldCrossValidation(trainData, trainLabel, outputAdr_validation, **kwargs):
    '''
    Train and validate the SVM classifier with n-fold cross validation and evaluate the performance
    input:
    trainData [[int]]: feature vector
    trainLabel [int]: labels
    outputAdr_validation str: address of output file
    return:
    None
    '''
    # set default parameters
    n_target = 4 if 'n_target' not in kwargs else kwargs['n_target'] # n_target is the number of label that need to be predicted. 
    n_feature = 21 if 'n_feature' not in kwargs else kwargs['n_feature']
    samplingMethod = 0 if 'samplingMethod' not in kwargs else kwargs['samplingMethod'] # samplingMethod=0: stratified sampling; samplingMethod=1:down sampling; samplingMethod=2: over sampling
    predictedLabel = [0,3] if 'predictedLabel' not in kwargs else kwargs['predictedLabel']
    label_dict = {0:'unacc', 1:'acc', 2:'good', 3:'vgood'}
    sampling = {0:'Stratified Sampling', 1:'Down Sampling', 2:'Over Sampling'}
    # load label and data
    data = [trainData, trainLabel]
    # down sampling or over sampling or do nothing 
    print('%s is chosen' %sampling[samplingMethod]) 
    if samplingMethod == 0:
        x = data[0]; y =data[1]  
    elif samplingMethod == 1:
        ros = RandomUnderSampler()
        x, y = ros.fit_sample(data[0], data[1])
    elif samplingMethod == 2:
        ros = RandomOverSampler()
        x, y = ros.fit_sample(data[0], data[1])
    print("After %s, the number of samples in each category is" %sampling[samplingMethod], sorted(Counter(y).items()))
        
    # predict with SVM
    clf = svm.SVC(kernel='linear')
    
    predicted_y = cross_val_predict(clf, x, y, cv=5) # 5-fold cross validation
    c_matrix = confusion_matrix(y, predicted_y, labels=list(range(predictedLabel[0], predictedLabel[1]+1))); 
    print('5-fold cross validation is done')
    print("Accuracy is", accuracy_score(y, predicted_y), 'Confusion Matrix is'); print(c_matrix)
    print(classification_report(y, predicted_y, labels=list(range(predictedLabel[0], predictedLabel[1]+1))))
    
    
    #Output the true label along side with the predicted label for each data item
    outputLabel = list( zip([label_dict[i] for i in trainLabel], [label_dict[i] for i in predicted_y]) )
    outputData = 'TrueLabel\tPredictLabel\n' + '\n'.join([x[0]+'\t'+x[1] for x in outputLabel])
    outputToTxt(outputAdr_validation, outputData)
        
def outputToTxt(outputAdr, outputData):
    # output data to file
    with open(outputAdr, 'w') as f:
        f.write(outputData)

    
if __name__ == '__main__':
	main()