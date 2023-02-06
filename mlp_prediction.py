#__importing libriries___________________________________________________________________________________________________________
import sys, os, math, csv
import numpy as np, pandas as pd
import sklearn, rdkit#, imblearn
import matplotlib, matplotlib.pyplot as plt
import tensorflow as tf

from rdkit import Chem
from rdkit import DataStructs

#from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split#, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn import metrics
from sklearn.utils import class_weight

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

#__ build dataset _______________________________________________________________________________________________________________
class DataSet:
    
    def __init__(self, pubchem_path_, all_se_path_):
        self.pubchem_path = pubchem_path_
        self.all_se_path = all_se_path_

    #__ creates dataframe with Prefered Term (PT) side effects from meddra_all_se.tsv ___________________________________________
    def FilterSEByType(self, type_):
        data_frame = pd.read_csv(self.all_se_path, sep='\t')

        #__ boolean vector of rows to consider (true if == term) __
        rows_to_consider = data_frame.values[:,3] == type_ 
        pt_df =  data_frame.loc[rows_to_consider, :]

        print('\nFiltering side effects by term: ' + type_ )

        return pt_df

    #__ returns dictionary with the keys being the filtered side effects and as values the number of occurences _________________
    def FilterSEByOcc(self, min_occ_, max_occ_, pt_df_, o_cut_):
        unique_se, counts = np.unique(pt_df_.values[:,4], return_counts = True)
        se_counts_dict = dict(zip(unique_se, counts))

        #__ dictionary comprehension __
        filtered_se = {unique_se:counts for unique_se, counts in se_counts_dict.items() if (counts > min_occ_ and counts < max_occ_)}

        print( '\nThe number of filtered side effects occurring more than '+str(min_occ_)+' times and less than '+str(max_occ_)+' is:' )
        print( len(filtered_se.keys()) )
        cut = len(unique_se)-len(filtered_se.keys())
        cutperc = cut/len(unique_se) * 100
        print(f'Cut from {len(unique_se)}: {cut} : {cutperc}%')
        with open(o_cut_, 'w') as f:
            f.write(f'Min occ. {min_occ_}\tMax occ. {max_occ_}\n')
            f.write(f'Number of original side effects: {len(unique_se)}\n')
            f.write(f'Number of filtered side effects: {len(filtered_se.keys())}\n')
            f.write(f'Cut: {cut}\tCut percentage: {cutperc}%\n')

        return filtered_se

    #__ Creates dictionary cid-side effects with only the filtered by occurences side effects ___________________________________
    def GetDict(self, filtered_se_, pt_df_):

        #__ in the all side effects file, the cid to consider is the STEREO CID, second column __
        #__ loop to remove "CID" string in file before integer identifier and casts to int ___
        cids = pt_df_.values[:, 1]
        for i in range( len(cids) ):
            cids[i] = cids[i][3:]
            cids[i] = int(cids[i])

        #__ populate dictionary __

        CID_SE = dict()
        
        for index, row in pt_df_.iterrows():

            #__ get cid of current row __
            current_cid = row[1] # -> this gives me INTEGER values because previously change with cids CHANGED pt_df                             
            row_se = row[4]
            
            #__ put only side effects that ARE ALSO FOUND in filtered_se __
            if row_se in filtered_se_.keys():
                if current_cid not in CID_SE.keys():
                    CID_SE[current_cid] = []
                CID_SE[current_cid].append(row_se)

        #__ check __
        print("\nThe number of drugs is:")
        print( len(CID_SE.keys()) )

        return CID_SE

    #__ function returns FEATURES data frame ____________________________________________________________________________________
    def GetFeatures(self, CID_SE_, fpSize_):
        df = pd.read_csv( self.pubchem_path, index_col = ['cid'], 
                          usecols = ['cid', 'mw', 'polararea', 'xlogp', 'heavycnt', 
                                    'hbonddonor', 'hbondacc', 'rotbonds', 'isosmiles'] )

        #__ include only the rows in the df dataframe that have a 'cid' value
        #   which is also found in CID_SE_.keys() __
        df = df[df.index.isin(CID_SE_.keys())]

        #__ set null features to 0 __
        df.fillna(0, inplace=True)

        print("\n\nFeatures:\n")
        print(df)

        #__ iterate over rows to extract fingerprints __
        for index, row in df.iterrows():
            smiles = row['isosmiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdkit.Chem.rdmolops.RDKFingerprint(mol, fpSize = fpSize_)
                
                #__ Transforming RDKit ExplicitBitVect() into binary numpy array __
                binary = np.zeros(len(fp), dtype = float)
                for i in range( len(fp) ):
                    if fp[i] > 0:
                        binary[i] = 1
                df.at[index, 'isosmiles'] = binary
                
            else:
                print(f'Invalid SMILES: {smiles}')
                
        #__ rename isosmiles column to fingerprints column __
        df.rename( columns = {'isosmiles': 'binary_fingerprints'}, inplace = True)
        
        #__ check __
        print("\n\nFeatures with derived fingerpints:\n")
        print(df)

        return df
    
   #__ checks consistency of data and deletes rows of the identifiers that aren't found both in features and targets ____________
   #__ Also orders the drugs in the same order for both features and targets ____________________________________________________
    def Consistency(self, cid_se_dict_, cid_feat_df_):

        #__ stereo-cids need to coincide __
        se_sorted = dict(sorted(cid_se_dict_.items() ) )
        feat_sorted = cid_feat_df_.sort_index()

        keys = list(se_sorted.keys())
        indexes = list(feat_sorted.index)

        rmvd_drugs = []

        i = 0
        while i < len(indexes):
            if indexes[i] not in keys:
                rmvd_drugs.append( indexes[i] )
                del feat_sorted[ indexes[i] ]

            i += 1

        i = 0
        while i < len(keys):
            if keys[i] not in indexes:
                rmvd_drugs.append( keys[i] )
                del se_sorted[ keys[i] ]
            
            i += 1

        print(f'\nRemoved drugs: \n{rmvd_drugs}')
        
        #__ check __
        new_keys = list(se_sorted.keys())
        new_indexes = list(feat_sorted.index)
        print("\nEvaluating data consistency:")
        print("Cid SE rows: %d" % len(se_sorted.keys() ) )
        print("Cid Feat. rows: %d" % feat_sorted.shape[0] )

        if( new_keys != new_indexes ):
            print("\nDrugs in features and targets aren't the same!")

        return se_sorted, feat_sorted

    #__ dummies table -> columns are side effects, iterate over rows for drugs ___________________________________________________
    def GetDummies(self, CID_SE_, filtered_se_):
        
        #__ Create unique side effects columns.
        #   Populate with ones only SE of corresponding drug __
        all_filtered_side_effects = list( filtered_se_.keys() )

        df = pd.DataFrame(0., index=CID_SE_.keys(), columns = all_filtered_side_effects )

        for drug in CID_SE_.keys():
            for side_effect in CID_SE_[drug]:
                df.loc[drug, side_effect] = 1.

        #__ check __
        print("\nThe number of columns in Dummies table is: %d\n" %df.shape[1])
        print(f'\nSorted side effects in dummies:\n{df}')
        
        check = df
        y = df.to_numpy()

        return y, check
        
    #__ scale and return features in numpy ______________________________________________________________________________________
    def ScaleAndVectorizeFeatures(self, features_, only_fp_):
        features1 = features_.iloc[:, :7].to_numpy()

        #__ row_stack() expands fingerprint to fit each value in a separate column __
        fingerprints = np.row_stack( features_.iloc[:, 7] )

        #__ if True returns only fp array __
        if only_fp_ == False:
            scaler1 = RobustScaler()
            scaler2 = MinMaxScaler()
            features1 = scaler1.fit_transform( features1 )
            features1 = scaler2.fit_transform( features1 )
            print("\nScaled features:")
            print(features1)
            print(features1.shape)
            print("\nFingerprints array:")
            print(fingerprints)
            print(fingerprints.shape)
            X_normalized = np.concatenate( (features1, fingerprints), axis = 1 )
            print("\nConcatenated Feature Set:")
            print(X_normalized)
            print(X_normalized.shape)

            return X_normalized

        return fingerprints
    
    def PrintDataFrame(self, df_, path_):
        if type(df_) is dict:
            df_ = pd.DataFrame.from_dict(df_, orient = 'index')
        elif type(df_) is np.ndarray:
            df_ = pd.DataFrame(df_)

        df_.to_csv(path_)


#__ machine learning model ______________________________________________________________________________________________________
class MultiLabelClassifier():

    def __init__(self, X_, y_):
        self.X = X_
        self.y = y_

    #__ splits data in training, testing and validation sets ____________________________________________________________________
    def SplitAndWeight(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 112)#, stratify = self.y)#stratify mantains class distribution but not compatible
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.3, random_state = 112)#, stratify = self.y)#stratify mantains class distribution but not compatible

        #__ check which drug has the most side effects in the training set __
        most1 = []
        for i in range(y_train.shape[0]):
            most1.append( (y_train[i] == 1).sum() )
        print(f'\nSample with most ones: {np.argmax(most1)} with {most1[np.argmax(most1)]} ones')

        #__ scikit-learn approach: ‘balanced’ class weights will be given by n_samples / (n_classes * np.bincount(y)) __
        #__ here denominator it's inverted to give more weight to samples with higher count of ones __
        sample_weights = np.ones(y_train.shape[0])
        for i in  range( y_train.shape[0] ):
            num_pos = (y_train[i, :] == 1).sum()

            #__ gives more weight based on frequency of ones. Than rescales by n_samples __
            sample_weights[i] = ( (y_train.shape[0]) * (num_pos/y_train.shape[1]) )
        
        class_weights = []
        for i in  range( y_train.shape[1] ):
            cw = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train[:, i] ), y = y_train[:, i] )
            class_weight_dict = dict(enumerate( cw ))
            class_weights.append( class_weight_dict )
                
        sample_weights_dict = dict(enumerate(sample_weights))
        class_weights_dict = dict(enumerate(class_weights))
        print("\nSample Weights:")
        print(sample_weights_dict)

        '''
        #__ multilabel not supported __
        ros = imblearn.over_sampling.RandomOverSampler( sampling_strategy = 0.5)
        rus = imblearn.under_sampling.RandomUnderSampler( sampling_strategy = 0.5)
        
        X_resamp, y_resamp = ros.fit_resample(X_train, y_train)
        X_resamp, y_resamp = rus.fit_resample(X_resamp, y_resamp)
        '''
        
        print("\nTraining set samples:\tX -> (%d, %d) \ty -> (%d, %d)" 
                %(X_train.shape[0], X_train.shape[1], y_train.shape[0], y_train.shape[1]) )
        print("Test set samples:\tX -> (%d, %d) \ty -> (%d, %d)" 
                %(X_test.shape[0], X_test.shape[1], y_test.shape[0], y_test.shape[1]) ) 
        print("Validation set samples:\tX -> (%d, %d) \ty -> (%d, %d) \n" 
                %(X_val.shape[0], X_val.shape[1], y_val.shape[0], y_val.shape[1]) )
        
        return  X_train, X_test, X_val, y_train, y_test, y_val, sample_weights, class_weights_dict

    #__ tunes hypermarameters and gets scores ___________________________________________________________________________________
    def ModelEvaluation(self, X_train, X_test, X_val, y_train, y_test, y_val, sweights_, cweights_, o_scores_, o_pr_, o_roc_):
        
        #__ Define hyperparameters __
        hiddL_sizes = [ (20,), (30,), (50,), (75,), (100, ), (150,), 
                        (200,), (300,), (400,), (500,), (600,), (700,), (800,), (1000,), (1200,)
                        #(1000,), (400, 300, 300), (333, 333, 333), 
                        #(800, ), (400, 400),
                        #(300,), (100, 100, 100), (500, ), (150, 150, 150),
                        #(35, 35, 35), (50, 50), (50, 50, 50), (50, 50, 50, 50, 50, 50), (70, 70), 
                        #(75, 75), (75, 75, 75, 75),  
                        #(100, 70, 30), (100, 100, 100), (150, 150), 
                        #(200, 100), (200, 70, 30), (150, 70, 50, 30), (100, 70, 50, 50, 30),
                        #(300, 100, 300), (200, 200, 200), (200, 100, 200), 
                        #(100, 50, 50, 100)
                        #(150, ), (300,), (500, ), (800,), (1000, ), (1200,), (3000,)
                        #(10, ), (10, 10), (10, 10, 10), (10, 10, 10, 10),
                        #(50, ), (50, 50), (50, 50, 50), (50, 50, 50, 50),
                        #(100,), (100, 100), (100, 100, 100), (100, 100, 100, 100),
                        #(150, ), (150, 150), (150, 150, 150), (150, 150, 150, 150),
                        #(200,), (200, 200), (200, 200, 200), (200, 200, 200, 200),
                        #(300, ), (300, 300), (300, 300, 300), (300, 300, 300, 300),
                        #(400, ), (400, 400), (400, 400, 400), (400, 400, 400, 100),
                        #(500, ), (500, 500), (500, 500, 300), 
                        #(800, ), (800, 500), 
                        #(1000, ), (1000, 300)
                        #(1200, ), (1200, 100)
                        #(1300, )
                    ]

        activations = ['sigmoid']#'logistic', 'tanh', 'relu' --> KERAS: logistic = sigmoid
        solvers = ['adam']#!'rprop'!, 'sgd', 'adam', 'lbfgs' -> !rprop not possible, no lbfgs for keras
        learning_rates = ['constant']#'constant', 'invscaling', 'adaptive'
        max_iters = [100, 200, 300, 400, 500, 600, 800, 1000]#100, 200, 300, 400, 500, 600, 800, 1000, 2000, 3000
        lr_inits = [0.0001]#0.001, 0.0005, 0.0001, 0.00005, 0.00001
        thresholds = [(0.5),]#0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.001

        best_scores1 = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                         'average precision': 0.0, 'auc': 0.0
        }

        best_params1 = { 'hidden_layer_sizes': None, 'activation': None, 'solver': None,
                        'learning_rate': None, 'learning_rate_init': None, 
                        'max_iter': None
        }

        best_scores2 = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                         'average precision': 0.0, 'auc': 0.0
        }

        best_params2 = { 'hidden_layer_sizes': None, 'activation': None, 'solver': None,
                        'learning_rate': None, 'learning_rate_init': None, 
                        'max_iter': None
        }

        best_threshold1 = None
        best_threshold2 = None

        position1 = None
        position2 = None

        evaluation1 = 'average precision'
        evaluation2 = 'auc'

        count = -1

        #__ clear file to append later __
        with open(o_scores_, 'w') as f:
            f.write(" ")

        #__ Loop over all combinations of hyperparameters __
        for hls in hiddL_sizes:
            for activ_fun in activations:
                for solver_ in solvers:
                    for lr in learning_rates:
                        for epochs in max_iters:
                            for init in lr_inits:
                                for threshold in thresholds:

                                    count += 1
                                    print("#%d" %count)
                                    
                                    #__ Create an instance of MLPClassifier with the current combination of hyperparameters __
                                    current_params = { 'hidden_layer_sizes': hls, 'activation': activ_fun, 'solver': solver_,
                                                        'learning_rate': lr, 'learning_rate_init': init, 
                                                        'max_iter': epochs
                                    }

                                    #model = MLPClassifier(**current_params, batch_size = 64, random_state = 42)

                                    #__ Create keras sequential layers. 
                                    #   Last layer activation function is always sigmoid __
                                    model = Sequential()

                                    for i, layer in enumerate(current_params['hidden_layer_sizes'] ):
                                        if i == 0:
                                            model.add( Dense(units = layer, activation = current_params['activation'], input_shape = (X_train.shape[1], ) ) )
                                        else:
                                            model.add( Dense(units = layer, activation = current_params['activation']) )

                                        #__ add last layer with units as number of target classes __
                                        model.add( Dense( units = y_train.shape[1], activation = 'sigmoid' ) )#'softmax')) --> not softmax because labels are independent.

                                    #__ define custom weighted binary cross entropy loss --> not used __
                                    def weightedloss(y_true, y_pred):
                                        loss = 0.0
                                        for index, weights in cweights_.items(): #weights_ is a dict of dicts. First keys = index = position. Second dict for zeroes and ones.
                                            #for cl in y_true[index]: #select column class in row
                                                loss -= ( ( weights[1] * y_true[index] * keras.backend.log(y_pred[index]) ) + ( weights[0] * (1 - y_true[index]) * keras.backend.log(1-y_pred[index]) ) )

                                        return keras.backend.mean(loss)

                                    if solver_ == 'adam':
                                        model.compile( optimizer = Adam(learning_rate = current_params['learning_rate_init']), loss = 'binary_crossentropy' )
                                    elif solver_ == 'sgd':
                                        model.compile( optimizer = SGD(learning_rate = current_params['learning_rate_init']), loss = 'binary_crossentropy' )
                                    else:
                                        model.compile( loss = 'binary_crossentropy' )

                                    #model.fit(X_train, y_train)
                                    #model.fit(X_train, y_train, epochs = current_params['max_iter'], batch_size = 64)
                                    model.fit(X_train, y_train, sample_weight = sweights_, epochs = current_params['max_iter'], batch_size = 64)

                                    #__ to which threshold setting value to 1 __
                                    y_proba = model.predict(X_val)
                                    y_pred = (y_proba >= threshold).astype(int)

                                    print(f'\nPred. Sizes: {y_pred.shape}')
                                    print(y_proba)

                                    precision = metrics.precision_score(y_val, y_pred, average = 'micro')
                                    recall = metrics.recall_score(y_val, y_pred, average = 'micro')
                                    f1 = metrics.f1_score(y_val, y_pred, average = 'micro')

                                    #__ providig probability estimates __
                                    avg_prec = metrics.average_precision_score(y_val, y_proba, average = 'micro')
                                    auc = metrics.roc_auc_score(y_val, y_proba, average = 'micro')
                                    
                                    #__ tn, fp, fn, tp __
                                    cm = metrics.multilabel_confusion_matrix(y_val, y_pred, samplewise = True)
                                    print(cm)

                                    current_scores = { 'f1': f1, 'precision': precision, 'recall': recall,
                                                     'average precision': avg_prec ,'auc': auc}

                                    print(f'\nParameters set to {current_params}, threshold: {threshold}')
                                    print(f'\nScores: {current_scores}\n')
                                    #print(metrics.classification_report(y_val, y_pred))

                                    #__ write to file current values __
                                    f = open(o_scores_, 'a')
                                    f.write(f'\n\n\n#{count}')
                                    f.write(f'\nParameters: {current_params}')
                                    f.write(f'\nScores: {current_scores}')
                                    f.close()

                                    #__ Plot the micro-averaged Precision-Recall curve __
                                    precision_micro, recall_micro, thpr = metrics.precision_recall_curve(y_val.ravel(), y_proba.ravel() )
                                    displaypr = metrics.PrecisionRecallDisplay( precision = precision_micro, recall = recall_micro, 
                                                                                average_precision = current_scores['average precision'] )

                                    displaypr.plot()
                                    plt.savefig(o_pr_+'pr'+str(count)+'.png')

                                    #__ Plot micro-averaged Precision-Recall curve __
                                    fpr_, tpr_, throc = metrics.roc_curve(y_val.ravel(), y_proba.ravel())
                                    displayroc = metrics.RocCurveDisplay( fpr = fpr_, tpr = tpr_, roc_auc = auc)

                                    displayroc.plot()
                                    plt.savefig(o_roc_+'roc'+str(count)+'.png')

                                    #__ Update the best combination of hyperparameters __
                                    if current_scores[evaluation1] > best_scores1[evaluation1]:
                                        best_params1.update(current_params)
                                        best_scores1.update(current_scores)
                                        best_threshold1 = float(threshold)
                                        position1 = int(count)

                                    if current_scores[evaluation2] > best_scores2[evaluation2]:
                                        best_params2.update(current_params)
                                        best_scores2.update(current_scores)
                                        best_threshold2 = float(threshold)
                                        position2 = int(count)

        #__ print to txt __
        with open(o_scores_, 'a') as f:

            f.write(f'\n\nBest scores based on {evaluation1}: ')
            for key, value in best_scores1.items():
                f.write(f' #{position1}-{key}: {value}\t')
            
            f.write(f"\n\nBest parameters for {evaluation1}: ")
            for key, value in best_params1.items():
                f.write(f' #{position1}-{key}: {value}\t')

            f.write(f'\n\nBest scores based on {evaluation2}: ')
            for key, value in best_scores2.items():
                f.write(f' #{position2}-{key}: {value}\t')
            
            f.write(f"\n\nBest parameters for {evaluation2}: ")
            for key, value in best_params2.items():
                f.write(f' #{position2}-{key}: {value}\t')

            f.write("\n\n\n\nSample Weights: ")
            f.write( str(sweights_) )
            
        print(f'\nBest hyperparameters based on {evaluation1}: {best_params1}, threshold: {best_threshold1}')
        print(f'\nBest scores based on {evaluation1}: {best_scores1}')

        print(f'\nBest hyperparameters based on {evaluation2}: {best_params2}, threshold: {best_threshold2}')
        print(f'\nBest scores based on {evaluation2}: {best_scores2}')

        if best_scores1[evaluation1] != 0.0:
            #model = MLPClassifier(**best_params1, batch_size = 64, random_state = 42)
            #model.fit(X_train, y_train)
            #y_pred = model.predict(X_test)
            with open(o_scores_, 'a') as f:
                f.write(f'\n\n\nFinal evaluation for best model by {evaluation1} on test set')

            for i, layer in enumerate(best_params1['hidden_layer_sizes'] ):
                if i == 0:
                    model.add( Dense(units = layer, activation = best_params1['activation'], input_shape = (X_train.shape[1], ) ) )
                else:
                    model.add( Dense(units = layer, activation = best_params1['activation']) )

                #__ add last layer with units as number of target classes __
                model.add( Dense( units = y_train.shape[1], activation = 'sigmoid' ) )#'softmax'))

            if best_params1['solver'] == 'adam':
                model.compile( optimizer = Adam(learning_rate = best_params1['learning_rate_init']), loss = 'binary_crossentropy' )
            elif best_params1['solver'] == 'sgd':
                model.compile( optimizer = SGD(learning_rate = best_params1['learning_rate_init']), loss = 'binary_crossentropy' )
            else:
                model.compile( loss = 'binary_crossentropy' )

            #model.fit(X_train, y_train, epochs = best_params1['max_iter'], batch_size = 64)
            model.fit(X_train, y_train, sample_weight = sweights_, epochs = best_params1['max_iter'], batch_size = 64)

            #__ valuate threshold based on f1 __
            fthresholds = [ 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1 ]
            bs = { 'threshold': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0 }

            for th in fthresholds:
                y_proba = model.predict(X_test)
                y_pred = (y_proba >= th).astype(int)
                precision = metrics.precision_score(y_test, y_pred, average = 'micro')
                recall = metrics.recall_score(y_test, y_pred, average = 'micro')
                f1 = metrics.f1_score(y_test, y_pred, average = 'micro')
                cm = metrics.multilabel_confusion_matrix(y_test, y_pred, samplewise = True)
                print(cm)

                #__ providig probability estimates __
                avg_prec = metrics.average_precision_score(y_test, y_proba, average = 'micro')
                auc = metrics.roc_auc_score(y_test, y_proba, average = 'micro')

                current_scores = { 'f1': f1, 'precision': precision, 'recall': recall,
                                    'average precision': avg_prec ,'auc': auc}

                with open(o_scores_, 'a') as f:
                    f.write(f'\n\nThreshold {th}, scores: {current_scores}')

                if f1 > bs['f1']:
                    bs['threshold'] = (th)
                    bs['f1'] = f1
                    bs['recall'] = recall
                    bs['precision'] = precision

            t = bs['threshold']
            print(f'\nBest threshold: {t}')     
            with open(o_scores_, 'a') as f:
                f.write(f'\n\nOptimizing f1 with threshold: {t}')


        if best_scores2[evaluation2] != 0.0:
            with open(o_scores_, 'a') as f:
                f.write(f'\n\n\nFinal evaluation for best model by {evaluation2} on test set')
            for i, layer in enumerate(best_params2['hidden_layer_sizes'] ):
                if i == 0:
                    model.add( Dense(units = layer, activation = best_params2['activation'], input_shape = (X_train.shape[1], ) ) )
                else:
                    model.add( Dense(units = layer, activation = best_params2['activation']) )

                #__ add last layer with units as number of target classes __
                model.add( Dense( units = y_train.shape[1], activation = 'sigmoid' ) )#'softmax'))

            if best_params2['solver'] == 'adam':
                model.compile( optimizer = Adam(learning_rate = best_params2['learning_rate_init']), loss = 'binary_crossentropy' )
            elif best_params2['solver'] == 'sgd':
                model.compile( optimizer = SGD(learning_rate = best_params2['learning_rate_init']), loss = 'binary_crossentropy' )
            else:
                model.compile( loss = 'binary_crossentropy' )

            #model.fit(X_train, y_train, epochs = best_params2['max_iter'], batch_size = 64)
            model.fit(X_train, y_train, sample_weight = sweights_, epochs = best_params2['max_iter'], batch_size = 64)#

            #__ valuate threshold based on f1 __
            fthresholds = [ 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1 ]
            bs = { 'threshold': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0 }

            for th in fthresholds:
                y_proba = model.predict(X_test)
                y_pred = (y_proba >= th).astype(int)
                precision = metrics.precision_score(y_test, y_pred, average = 'micro')
                recall = metrics.recall_score(y_test, y_pred, average = 'micro')
                f1 = metrics.f1_score(y_test, y_pred, average = 'micro')
                cm = metrics.multilabel_confusion_matrix(y_test, y_pred, samplewise = True)
                print(cm)
                
                #__ providig probability estimates __
                avg_prec = metrics.average_precision_score(y_test, y_proba, average = 'micro')
                auc = metrics.roc_auc_score(y_test, y_proba, average = 'micro')
                current_scores = { 'f1': f1, 'precision': precision, 'recall': recall,
                                    'average precision': avg_prec ,'auc': auc }

                with open(o_scores_, 'a') as f:
                    f.write(f'\n\nThreshold {th}, scores: {current_scores}')

                if f1 > bs['f1']:
                    bs['threshold'] = (th)
                    bs['f1'] = f1
                    bs['recall'] = recall
                    bs['precision'] = precision

            t = bs['threshold']
            print(f'\nBest threshold: {t}')     
            with open(o_scores_, 'a') as f:
                f.write(f'\n\nOptimizing f1 with threshold: {t}')


#__ main function _______________________________________________________________________________________________________________
def main():

    matplotlib.use('Agg')

    #__ get the current directory and define paths/variables __
    current_directory = os.path.dirname(os.path.abspath(__file__))
    pbc_path = os.path.join(current_directory, 'Data/PubChem_compound_stereocid.csv')
    se_path = os.path.join(current_directory, 'Data/SIDER/meddra_all_se.tsv')
    o_cut = os.path.join(current_directory, 'Output/filtered_side_effects_info.txt')
    o_se = os.path.join(current_directory, 'Output/output_se.csv')
    o_feat = os.path.join(current_directory, 'Output/output_feat.csv')
    o_scaled = os.path.join(current_directory, 'Output/output_scaled_feat.csv')
    o_dummies = os.path.join(current_directory, 'Output/output_dummies.csv')
    o_scores = os.path.join(current_directory, 'Output/output_scores.txt')
    o_pr = os.path.join(current_directory, 'Output/PR Curve Plots/')
    o_roc = os.path.join(current_directory, 'Output/ROC Curve Plots/')

    #__ filter side effects by term and occurencies __
    meddra_type = 'PT'      #MedDRA concept type. Prefered Term (PT) or Lowest Level Term (LLT)
    min_occ = 259            #Min occurancies of a side effect for acceptance
    max_occ = 2000          #Max occurancies of a side effect for acceptance

    #__ features options __
    fp_size = 2048          #Default fingerprint size is 2048
    get_only_fp = False     #Get only fingerpints for features or also PubChem descriptors?


    #__ initiate class Dataset and call class functions __
    dataset = DataSet(pbc_path, se_path)
    pt_df = dataset.FilterSEByType(meddra_type)
    filtered_se = dataset.FilterSEByOcc(min_occ, max_occ, pt_df, o_cut)
    drugs_se_dict = dataset.GetDict(filtered_se, pt_df)
    features_df = dataset.GetFeatures(drugs_se_dict, fp_size)
    drugs_se_dict, features_df = dataset.Consistency(drugs_se_dict, features_df)

    #__ print to file __
    dataset.PrintDataFrame(drugs_se_dict, o_se)
    dataset.PrintDataFrame(features_df, o_feat)

    y, check = dataset.GetDummies(drugs_se_dict, filtered_se)

    #__ print to file __
    dataset.PrintDataFrame(check, o_dummies)

    X = dataset.ScaleAndVectorizeFeatures(features_df, get_only_fp)

    #__ print to file __
    dataset.PrintDataFrame(X, o_scaled)
    

    #__ initiate class MultiLabelClassifier and call class functions __
    classifier = MultiLabelClassifier(X, y)
    X_train, X_test, X_val, y_train, y_test, y_val, sweights, cweights = classifier.SplitAndWeight()
    classifier.ModelEvaluation(X_train, X_test, X_val, y_train, y_test, y_val, sweights, cweights, o_scores, o_pr, o_roc)


#__executes main only if __name__ special variable is equal to "__main__"__________________________________________________
if __name__ == "__main__":
    main()
        
